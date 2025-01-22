import { lstatSync, readFileSync, readdirSync } from "fs";
import * as path from "path";
import { Err, Ok } from "ts-results";
import { Position } from "vscode-languageclient";

import { createCoqLspClient } from "../coqLsp/coqLspBuilders";
import { CoqLspClient, DocumentSpec } from "../coqLsp/coqLspClient";

import {
    CoqAugmentedTheoremItem,
    CoqDataset,
    CoqDatasetAugmentedFile,
    CoqDatasetFolder,
    CoqDatasetTheoremItem,
    NodeAugmentationResult,
} from "../coqDatasetRepresentation/coqDatasetModels";
import {
    generateFileViewer,
    generateFolderViewer,
} from "../coqDatasetRepresentation/generateDatasetViewer";
import {
    accumulateStats,
    createTheoremWithStats,
    emptyDatasetStats,
} from "../coqDatasetRepresentation/statisticsCalculation";
import { parseCoqFile } from "../coqParser/parseCoqFile";
import { buildCoqProofTree } from "../coqProofTree/buildCoqProofTree";
import { CoqTheoremValidator } from "../coqProofTree/coqTheoremValidator";
import { augmentTreeToSamples } from "../coqProofTree/proofBuilder";
import { EventLogger, Severity } from "../logging/eventLogger";
import Logger from "../logging/logger";
import { getProgressBar } from "../logging/progressBar";
import { Uri } from "../utils/uri";

import { RunParams } from "./utilityRunParams";

export class ProjectProcessor {
    private constructor(
        public readonly eventLogger: EventLogger,
        public readonly logger: Logger,
        public readonly abortController: AbortController,
        public readonly runArgs: RunParams,
        private readonly coqLspClient: CoqLspClient
    ) {}

    static async create(runArgs: RunParams): Promise<ProjectProcessor> {
        const eventLogger: EventLogger = new EventLogger();
        const logger: Logger = new Logger(eventLogger, runArgs.loggingLevel);
        const abortController = new AbortController();

        const coqLspClient = await createCoqLspClient(
            runArgs.workspaceRootPath,
            runArgs.coqLspServerPath,
            undefined,
            eventLogger,
            abortController
        );

        return new ProjectProcessor(
            eventLogger,
            logger,
            abortController,
            runArgs,
            coqLspClient
        );
    }

    async processProject(): Promise<CoqDataset> {
        if (this.isCoqFile(this.runArgs.targetRootPath)) {
            return this.processFile(
                this.runArgs.targetRootPath,
                this.runArgs.targetRootPath
            );
        } else if (lstatSync(this.runArgs.targetRootPath).isDirectory()) {
            // In this case collectedDataset is updated from inside the
            // function to gradually save progress (dumb way tho) in case of occured
            // errors
            return this.processDir(
                this.runArgs.targetRootPath,
                this.runArgs.targetRootPath,
                this.runArgs.generateDatasetViewer
            );
        } else {
            throw new Error(
                `Target path ${this.runArgs.targetRootPath} is not a Coq file or a directory`
            );
        }
    }

    private isCoqFile(filePath: string): boolean {
        return path.extname(filePath) === ".v";
    }

    private async processDir(
        rootPath: string,
        accumulatedPath: string,
        generateDatasetViewer: boolean = true
    ): Promise<CoqDatasetFolder> {
        const datasetFolderItem: CoqDatasetFolder = {
            dirPath: accumulatedPath,
            itemName: path.basename(accumulatedPath),
            stats: emptyDatasetStats(),
            dirItems: [],
            type: "dir",
        };

        try {
            const dirItemsAll = readdirSync(accumulatedPath, {
                withFileTypes: true,
                recursive: false,
            });
            const dirItems = dirItemsAll.filter(
                (item) =>
                    item.isDirectory() ||
                    (item.isFile() && item.name.endsWith(".v"))
            );

            for (const item of dirItems) {
                if (item.isDirectory()) {
                    await this.processDir(
                        rootPath,
                        `${accumulatedPath}/${item.name}`,
                        generateDatasetViewer
                    );
                } else if (item.isFile() && item.name.endsWith(".v")) {
                    const datasetItem = await this.processFile(
                        rootPath,
                        `${accumulatedPath}/${item.name}`,
                        generateDatasetViewer
                    );

                    datasetFolderItem.dirItems.push(datasetItem);
                    datasetFolderItem.stats = accumulateStats(
                        datasetFolderItem.stats,
                        datasetItem.stats
                    );
                }
            }
        } catch (e) {
            this.eventLogger.log(
                "error-processing-folder",
                `Error processing folder ${rootPath}: ${e}`,
                null,
                Severity.ERROR
            );
        }

        if (generateDatasetViewer) {
            generateFolderViewer(rootPath, accumulatedPath, datasetFolderItem);
        }

        return datasetFolderItem;
    }

    // TODO: Refactor
    private async processFile(
        rootPathRelative: string,
        filePathRelative: string,
        createDatasetViewer: boolean = true
    ): Promise<CoqDatasetAugmentedFile> {
        const rootPath = path.resolve(rootPathRelative);
        const filePath = path.resolve(filePathRelative);

        this.eventLogger.log(
            "started-processing-file",
            `Processing file ${filePath}`
        );

        const fileUri = Uri.fromPath(filePath);
        const fileLines = readFileSync(fileUri.fsPath).toString().split("\n");

        const parentDir = path.dirname(filePath);
        const fileName = path.parse(path.basename(filePath)).name;
        const docSpec: DocumentSpec = {
            uri: fileUri,
            version: 1,
        };
        const theoremValidator = new CoqTheoremValidator(this.coqLspClient);
        const coqDatasetFileItem: CoqDatasetAugmentedFile = {
            filePath: filePath,
            itemName: fileName,
            initialFileContent: fileLines,
            augmentedTheorems: [],
            stats: emptyDatasetStats(),
            type: "file",
        };

        await this.coqLspClient.withTextDocument(docSpec, async (_) => {
            const parsedTheorems = await parseCoqFile(
                fileUri,
                this.coqLspClient,
                this.abortController.signal,
                false,
                this.eventLogger
            );

            const progress = getProgressBar(fileName, parsedTheorems.length);

            this.eventLogger.log(
                "finished-processing-file",
                `Retrieved following theorems from file ${filePath}: ${parsedTheorems
                    .map((theorem) => theorem.name)
                    .join(", ")}`
            );

            for (const theorem of parsedTheorems) {
                this.eventLogger.log(
                    "started-processing-theorem",
                    `Processing theorem ${theorem.name}`
                );

                const proofTree = await buildCoqProofTree(
                    theorem,
                    this.coqLspClient,
                    fileUri,
                    1
                );
                if (proofTree.err) {
                    this.eventLogger.log(
                        "error-processing-theorem",
                        `Error processing theorem ${theorem.name}: ${proofTree.val.message}`
                    );

                    const theoremItem: CoqDatasetTheoremItem =
                        createTheoremWithStats(
                            theorem,
                            filePath,
                            Err(new Error(proofTree.val.message))
                        );
                    coqDatasetFileItem.augmentedTheorems.push(theoremItem);
                    coqDatasetFileItem.stats = accumulateStats(
                        coqDatasetFileItem.stats,
                        theoremItem.stats
                    );
                } else {
                    const samples = augmentTreeToSamples(
                        proofTree.val,
                        theorem.name
                    );

                    const theoremStartPos = theorem.statement_range.start;
                    const contentPrefix = this.getTextBeforePosition(
                        fileLines,
                        theoremStartPos
                    );
                    const validationResults =
                        await theoremValidator.validateTheorems(
                            parentDir,
                            contentPrefix,
                            theoremStartPos,
                            samples,
                            this.runArgs.theoremValidationTimeoutMillis,
                            this.runArgs.fileTypeCheckingTimeoutMillis
                        );

                    const nodeAugmentationRes = new Map<
                        number,
                        NodeAugmentationResult
                    >();
                    samples.forEach((sample, nodeId) => {
                        const validationRes = validationResults.get(nodeId);
                        if (validationRes === undefined) {
                            throw new Error(
                                `No validation result for node ${nodeId}`
                            );
                        }

                        nodeAugmentationRes.set(
                            nodeId,
                            validationRes.isValid
                                ? Ok(sample)
                                : Err(new Error(validationRes.diagnostic))
                        );
                    });

                    nodeAugmentationRes.forEach((res, nodeId) => {
                        this.eventLogger.log(
                            "node-augmentation-result",
                            `Node ${nodeId} augmentation result: ${res}`
                        );
                    });

                    const coqAugmentedTheorem: CoqAugmentedTheoremItem = {
                        samples: nodeAugmentationRes,
                        proofTree: proofTree.val,
                    };

                    const theoremItem: CoqDatasetTheoremItem =
                        createTheoremWithStats(
                            theorem,
                            filePath,
                            Ok(coqAugmentedTheorem)
                        );

                    coqDatasetFileItem.augmentedTheorems.push(theoremItem);
                    coqDatasetFileItem.stats = accumulateStats(
                        coqDatasetFileItem.stats,
                        theoremItem.stats
                    );
                }

                progress.update();
            }
        });

        if (createDatasetViewer) {
            generateFileViewer(
                rootPath,
                coqDatasetFileItem,
                this.runArgs.generateAugmentedCoqFiles
            );
        }

        return coqDatasetFileItem;
    }

    private getTextBeforePosition(
        textLines: string[],
        position: Position
    ): string[] {
        const textPrefix = textLines.slice(0, position.line + 1);
        textPrefix[position.line] = textPrefix[position.line].slice(
            0,
            position.character
        );
        return textPrefix;
    }

    dispose(): void {
        this.logger.dispose();
    }
}
