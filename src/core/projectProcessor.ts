import { readFileSync, readdirSync, lstatSync } from "fs";
import * as path from "path";
import { Err, Ok } from "ts-results";
import { Position } from "vscode-languageclient";

import { createCoqLspClient } from "../coqLsp/coqLspBuilders";
import { CoqLspClient, DocumentSpec } from "../coqLsp/coqLspClient";

import {
    accumulateStats,
    CoqAugmentedTheoremItem,
    CoqDataset,
    CoqDatasetAugmentedFile,
    CoqDatasetFolder,
    emptyDatasetStats,
    NodeAugmentationResult,
} from "../coqDatasetRepresentation/coqDatasetModels";
import {
    generateFileViewer,
    generateFolderViewer,
} from "../coqDatasetRepresentation/generateDatasetViewer";
import { parseCoqFile } from "../coqParser/parseCoqFile";
import { buildCoqProofTree } from "../coqProofTree/buildCoqProofTree";
import { CoqTheoremValidator } from "../coqProofTree/coqTheoremValidator";
import { augmentTreeToSamples } from "../coqProofTree/proofBuilder";
import { EventLogger, Severity } from "../logging/eventLogger";
import Logger from "../logging/logger";
import { Uri } from "../utils/uri";

import { defaultUtilityRunParams } from "./utilityRunParams";
import { calculateSuccessfullyAugmentedNodes } from "../coqDatasetRepresentation/coqDatasetUtils";
import { getProgressBar } from "../logging/progressBar";

export class ProjectProcessor {
    private constructor(
        public readonly eventLogger: EventLogger,
        public readonly logger: Logger,
        public readonly abortController: AbortController,
        private readonly coqLspClient: CoqLspClient
    ) { }

    static async create(): Promise<ProjectProcessor> {
        const eventLogger: EventLogger = new EventLogger();
        const logger: Logger = new Logger(
            eventLogger,
            defaultUtilityRunParams.loggingLevel
        );
        const abortController = new AbortController();

        const coqLspClient = await createCoqLspClient(
            defaultUtilityRunParams.workspaceRootPath,
            defaultUtilityRunParams.coqLspServerPath,
            undefined,
            eventLogger,
            abortController
        );

        return new ProjectProcessor(
            eventLogger,
            logger,
            abortController,
            coqLspClient
        );
    }

    // TODO: Move rootPath param to function argument
    async processProject(): Promise<CoqDataset> {
        if (this.isCoqFile(defaultUtilityRunParams.targetRootPath)) {
            return this.processFile(
                defaultUtilityRunParams.targetRootPath,
                defaultUtilityRunParams.targetRootPath
            );
        } else if (lstatSync(defaultUtilityRunParams.targetRootPath).isDirectory()) {
            // In this case collectedDataset is updated from inside the
            // function to gradually save progress (dumb way tho) in case of occured 
            // errors
            return this.processDir(
                defaultUtilityRunParams.targetRootPath,
                defaultUtilityRunParams.targetRootPath
            );
        } else {
            throw new Error(
                `Target path ${defaultUtilityRunParams.targetRootPath} is not a Coq file or a directory`
            );
        }
    }

    private isCoqFile(filePath: string): boolean {
        return path.extname(filePath) === ".v";
    }

    private async processDir(
        rootPath: string,
        accumulatedPath: string,
        generateDatasetViewer: boolean = true,
        coqLspTimeoutMillis: number = 3000000
    ): Promise<CoqDatasetFolder> {
        const datasetFolderItem: CoqDatasetFolder = {
            dirPath: accumulatedPath,
            itemName: path.basename(accumulatedPath),
            stats: emptyDatasetStats(),
            dirItems: [],
            type: "dir",
        }

        try {
            const dirItemsAll = readdirSync(accumulatedPath, {
                withFileTypes: true,
                recursive: false,
            });
            const dirItems = dirItemsAll.filter((item) => item.isDirectory() || item.isFile() && item.name.endsWith(".v"));

            for (const item of dirItems) {
                if (item.isDirectory()) {
                    await this.processDir(
                        rootPath,
                        `${accumulatedPath}/${item.name}`,
                        generateDatasetViewer,
                        coqLspTimeoutMillis
                    );
                } else if (item.isFile() && item.name.endsWith(".v")) {
                    const datasetItem = await this.processFile(
                        rootPath,
                        `${accumulatedPath}/${item.name}`,
                        generateDatasetViewer,
                        coqLspTimeoutMillis
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
        createDatasetViewer: boolean = true,
        coqLspTimeoutMillis: number = 3000000
    ): Promise<CoqDatasetAugmentedFile> {
        const rootPath = path.resolve(rootPathRelative);
        const filePath = path.resolve(filePathRelative);

        this.eventLogger.log(
            "started-processing-file",
            `Processing file ${filePath}`
        );

        const fileUri = Uri.fromPath(filePath);
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

                    coqDatasetFileItem.augmentedTheorems.push({
                        parsedTheorem: theorem,
                        sourceFile: filePath,
                        proofTreeBuildResult: Err(
                            new Error(proofTree.val.message)
                        ),
                        augmentedNodesRatio: [0, 0],
                    });
                } else {
                    const samples = augmentTreeToSamples(
                        proofTree.val,
                        theorem.name
                    );

                    const theoremStartPos = theorem.statement_range.start;
                    const fileLines = readFileSync(fileUri.fsPath)
                        .toString()
                        .split("\n");

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
                            coqLspTimeoutMillis
                        );

                    const nodeAugmentationRes = new Map<number, NodeAugmentationResult>();
                    samples.forEach((sample, nodeId) => {
                        const validationRes = validationResults.get(nodeId);
                        if (validationRes === undefined) {
                            throw new Error(`No validation result for node ${nodeId}`);
                        }

                        nodeAugmentationRes.set(nodeId, validationRes.isValid ? Ok(sample) : Err(new Error(validationRes.diagnostic)));
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

                    const augmentedCount = calculateSuccessfullyAugmentedNodes(coqAugmentedTheorem);

                    coqDatasetFileItem.augmentedTheorems.push({
                        parsedTheorem: theorem,
                        sourceFile: filePath,
                        proofTreeBuildResult: Ok(coqAugmentedTheorem),
                        augmentedNodesRatio: augmentedCount
                    });

                    coqDatasetFileItem.stats.augmentedNodesRatio = [
                        coqDatasetFileItem.stats.augmentedNodesRatio[0] + augmentedCount[0],
                        coqDatasetFileItem.stats.augmentedNodesRatio[1] + augmentedCount[1],
                    ];
                }

                progress.update();
            }
        });

        if (createDatasetViewer) {
            generateFileViewer(rootPath, coqDatasetFileItem);
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
