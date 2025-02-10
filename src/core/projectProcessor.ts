import { lstatSync, readFileSync, readdirSync } from "fs";
import * as path from "path";
import { Err, Ok } from "ts-results";

import { createCoqLspClient } from "../coqLsp/coqLspBuilders";
import { CoqLspClient, DocumentSpec } from "../coqLsp/coqLspClient";

import {
    CoqDataset,
    CoqDatasetAugmentedFile,
    CoqDatasetFolder,
    CoqDatasetUnaugmentedFile,
    CoqDatasetUnvalidatedTheorem,
} from "../coqDatasetRepresentation/coqDatasetModels";
import { generateCoqAugmentedFile } from "../coqDatasetRepresentation/generateCoqAugmentedFile";
import {
    generateFileViewer,
    generateFolderViewer,
} from "../coqDatasetRepresentation/generateDatasetViewer";
import {
    accumulateStats,
    emptyDatasetStats,
} from "../coqDatasetRepresentation/statisticsCalculation";
import { parseCoqFile } from "../coqParser/parseCoqFile";
import { buildCoqProofTree } from "../coqProofTree/buildCoqProofTree";
import { CoqTheoremValidator } from "../coqProofTree/coqTheoremValidator";
import { EventLogger, Severity } from "../logging/eventLogger";
import Logger from "../logging/logger";
import { getProgressBar } from "../logging/progressBar";
import { Uri } from "../utils/uri";

import { RunParams } from "./utilityRunParams";
import { TimeMark } from "../coqProofTree/measureTimeUtils";

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
                    const dirItem = await this.processDir(
                        rootPath,
                        `${accumulatedPath}/${item.name}`
                    );

                    datasetFolderItem.dirItems.push(dirItem);
                    datasetFolderItem.stats = accumulateStats(
                        datasetFolderItem.stats,
                        dirItem.stats
                    );
                } else if (item.isFile() && item.name.endsWith(".v")) {
                    const datasetItem = await this.processFile(
                        rootPath,
                        `${accumulatedPath}/${item.name}`
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
        filePathRelative: string
    ): Promise<CoqDatasetAugmentedFile> {
        const rootPath = path.resolve(rootPathRelative);
        const filePath = path.resolve(filePathRelative);
        const timeMark = new TimeMark();

        this.eventLogger.log(
            "started-processing-file",
            `Processing file ${filePath}`
        );

        const fileUri = Uri.fromPath(filePath);
        const fileLines = readFileSync(fileUri.fsPath).toString().split("\n");

        const fileName = path.parse(path.basename(filePath)).name;
        const docSpec: DocumentSpec = {
            uri: fileUri,
            version: 1,
        };
        const coqDatasetUnaugmentedFile: CoqDatasetUnaugmentedFile = {
            filePath: filePath,
            itemName: fileName,
            initialFileContent: fileLines,
            theoremsWithProofTrees: [],
        };

        await this.coqLspClient.withTextDocument(docSpec, async (_) => {
            const parsedTheorems = await parseCoqFile(
                fileUri,
                this.coqLspClient,
                this.abortController.signal,
                false,
                this.eventLogger
            );

            const progress = getProgressBar(
                "Building proof trees for",
                fileName,
                parsedTheorems.length
            );

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

                    const theoremItem: CoqDatasetUnvalidatedTheorem = {
                        parsedTheorem: theorem,
                        sourceFilePath: filePath,
                        proofTreeBuildResult: Err(
                            new Error(proofTree.val.message)
                        ),
                    };
                    coqDatasetUnaugmentedFile.theoremsWithProofTrees.push(
                        theoremItem
                    );
                } else {
                    this.eventLogger.log(
                        "finished-processing-theorem",
                        `Finished processing theorem ${theorem.name}`
                    );

                    const theoremItem: CoqDatasetUnvalidatedTheorem = {
                        parsedTheorem: theorem,
                        sourceFilePath: filePath,
                        proofTreeBuildResult: Ok(proofTree.val),
                    };
                    coqDatasetUnaugmentedFile.theoremsWithProofTrees.push(
                        theoremItem
                    );
                }

                progress.update();
            }
        });

        const parentDir = path.dirname(filePath);
        const theoremValidator = new CoqTheoremValidator(
            this.abortController,
            this.runArgs,
            this.eventLogger
        );

        try {
            const coqDatasetFileItem =
                await theoremValidator.augmentFileWithValidSamples(
                    parentDir,
                    coqDatasetUnaugmentedFile,
                    this.runArgs.fileAugmentationTimeoutMillis,
                    this.runArgs.fileTypeCheckingTimeoutMillis
                );

            if (this.runArgs.generateDatasetViewer) {
                generateFileViewer(rootPath, coqDatasetFileItem);
            }

            if (this.runArgs.generateAugmentedCoqFiles) {
                const augmentedFileLength = generateCoqAugmentedFile(
                    rootPath,
                    coqDatasetFileItem,
                    this.runArgs.loggingLevel <= Severity.INFO
                );
                coqDatasetFileItem.stats.locChangeAfterAugmentation = [
                    fileLines.length,
                    augmentedFileLength,
                ];
                coqDatasetFileItem.stats.processingTimeMillis = timeMark.measureElapsedMillis();
            }

            return coqDatasetFileItem;
        } catch (e) {
            if (e instanceof Error) {
                this.eventLogger.log(
                    "error-processing-file",
                    `Error processing file ${filePath}: ${e.message}`,
                    null,
                    Severity.ERROR
                );

                let coqDatasetFileItem: CoqDatasetAugmentedFile = {
                    ...coqDatasetUnaugmentedFile,
                    augmentedTheorems: [],
                    stats: emptyDatasetStats(),
                    type: "file",
                };

                return coqDatasetFileItem;
            }
        }

        throw new Error("Unreachable code");
    }

    dispose(): void {
        this.logger.dispose();
    }
}
