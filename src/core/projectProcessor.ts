import { readFileSync, readdirSync } from "fs";
import * as path from "path";
import { Err, Ok } from "ts-results";
import { Position } from "vscode-languageclient";

import { createCoqLspClient } from "../coqLsp/coqLspBuilders";
import { CoqLspClient, DocumentSpec } from "../coqLsp/coqLspClient";
import {
    CoqAugmentedTheoremItem,
    CoqDataset,
    CoqDatasetAugmentedFile,
} from "../coqDatasetRepresentation/coqDatasetModels";
import { parseCoqFile } from "../coqParser/parseCoqFile";
import { buildCoqProofTree } from "../coqProofTree/buildCoqProofTree";
import { CoqTheoremValidator } from "../coqProofTree/coqTheoremValidator";
import { augmentTreeToSamples } from "../coqProofTree/proofBuilder";
import { EventLogger } from "../logging/eventLogger";
import Logger from "../logging/logger";
import { Uri } from "../utils/uri";

import { defaultUtilityRunParams } from "./utilityRunParams";
import { generateFileViewer, generateFolderViewer } from "../coqDatasetRepresentation/generateDatasetViewer";

export class ProjectProcessor {
    private collectedDataset: CoqDataset = [];

    get dataset(): CoqDataset {
        return this.collectedDataset;
    }

    private constructor(
        public readonly eventLogger: EventLogger,
        public readonly logger: Logger,
        public readonly abortController: AbortController,
        private readonly coqLspClient: CoqLspClient
    ) {}

    static async create(): Promise<ProjectProcessor> {
        const eventLogger: EventLogger = new EventLogger();
        const logger: Logger = new Logger(
            eventLogger,
            defaultUtilityRunParams.loggingLevel
        );
        const abortController = new AbortController();

        const coqLspClient = await createCoqLspClient(
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

    async processProject(
        path: string
    ) {
        await this.processDir(path, path);
    }

    private async processDir(
        rootPath: string,
        accumulatedPath: string,
        generateDatasetViewer: boolean = true,
        coqLspTimeoutMillis: number = 150000
    ) {
        try {
            const dirItems = readdirSync(accumulatedPath, { withFileTypes: true, recursive: false });

            if (generateDatasetViewer) {
                generateFolderViewer(rootPath, accumulatedPath, dirItems);
            }

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

                    this.collectedDataset.push(datasetItem);
                }
            }
        } catch(e) {
            this.eventLogger.log(
                "error-processing-file",
                `Error reading folder ${rootPath}: ${e}`
            );
        }
    }

    private async processFile(
        rootPath: string,
        filePath: string,
        createDatasetViewer: boolean = true,
        coqLspTimeoutMillis: number = 150000
    ): Promise<CoqDatasetAugmentedFile> {
        this.eventLogger.log(
            "started-processing-file",
            `Processing file ${filePath}`
        );

        const fileUri = Uri.fromPath(filePath);
        const parentDir = path.dirname(filePath);
        const docSpec: DocumentSpec = {
            uri: fileUri,
            version: 1,
        };
        const theoremValidator = new CoqTheoremValidator(this.coqLspClient);
        const coqDatasetFileItem: CoqDatasetAugmentedFile = {
            filePath: filePath,
            augmentedTheorems: [],
        };

        await this.coqLspClient.withTextDocument(docSpec, async (_) => {
            const parsedTheorems = await parseCoqFile(
                fileUri,
                this.coqLspClient,
                this.abortController.signal,
                false,
                this.eventLogger
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

                    coqDatasetFileItem.augmentedTheorems.push({
                        parsedTheorem: theorem,
                        sourceFile: filePath,
                        theoremAugmentationRes: Err(new Error(proofTree.val.message)),
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
                    const validationRes =
                        await theoremValidator.validateTheorems(
                            parentDir,
                            contentPrefix,
                            theoremStartPos,
                            samples,
                            coqLspTimeoutMillis
                        );

                    if (!validationRes.every((res) => res.isValid)) {
                        this.eventLogger.log(
                            "error-processing-theorem",
                            `Error processing theorem ${theorem.name}: ${validationRes
                                .filter((res) => !res.isValid)
                                .map((res) => res.diagnostic)
                                .join(", ")}`
                        );

                        coqDatasetFileItem.augmentedTheorems.push({
                            parsedTheorem: theorem,
                            sourceFile: filePath,
                            theoremAugmentationRes: Err(
                                new Error(
                                    validationRes
                                        .map((res) => res.diagnostic)
                                        .join(", ")
                                )
                            ),
                        });
                    } else {
                        const coqAugmentedTheorem: CoqAugmentedTheoremItem = {
                            samples: samples,
                            proofTree: proofTree.val,
                        };

                        coqDatasetFileItem.augmentedTheorems.push({
                            parsedTheorem: theorem,
                            sourceFile: filePath,
                            theoremAugmentationRes: Ok(coqAugmentedTheorem),
                        });
                    }
                }
            }
        })
        
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
