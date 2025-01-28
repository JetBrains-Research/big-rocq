import { Mutex } from "async-mutex";
import { appendFileSync, existsSync, unlinkSync, writeFileSync } from "fs";
import * as path from "path";

import { CoqLspClient } from "../coqLsp/coqLspClient";
import { CoqLspError, CoqLspTimeoutError } from "../coqLsp/coqLspTypes";

import { CoqAugmentedTheoremItem, CoqDatasetAugmentedFile, CoqDatasetTheoremItem, CoqDatasetUnaugmentedFile, NodeAugmentationResult, TheoremDatasetSample } from "../coqDatasetRepresentation/coqDatasetModels";
import { Uri } from "../utils/uri";

import { augmentTreeToSamples, theoremDatasetSampleToString } from "./proofBuilder";
import { accumulateTheoremStats, createTheoremWithStats, emptyDatasetStats } from "../coqDatasetRepresentation/statisticsCalculation";
import { getTextInRange } from "../utils/documentUtils";
import { Err, Ok } from "ts-results";
import { EventLogger } from "../logging/eventLogger";
import { getProgressBar } from "../logging/progressBar";

export interface TheoremValidationResult {
    theorem: TheoremDatasetSample;
    isValid: boolean;
    diagnostic?: string;
}

export interface CoqTheoremValidatorInterface {
    /**
     * To check that the pack of generated theorems is valid and the make sense.
     */
    augmentFileWithValidSamples(
        sourceDirPath: string,
        unaugmentedFile: CoqDatasetUnaugmentedFile
    ): Promise<CoqDatasetAugmentedFile>;

    dispose(): void;
}

export class CoqTheoremValidator implements CoqTheoremValidatorInterface {
    private mutex: Mutex = new Mutex();

    constructor(
        private readonly coqLspClient: CoqLspClient,
        public readonly eventLogger?: EventLogger
    ) {}

    async augmentFileWithValidSamples(
        sourceDirPath: string,
        unaugmentedFile: CoqDatasetUnaugmentedFile,
        fileAugmentationTimeoutMillis: number = 150000,
        fileTypeCheckingTimeoutMillis: number = 150000
    ): Promise<CoqDatasetAugmentedFile> {
        return await this.mutex.runExclusive(async () => {
            const timeoutPromise = new Promise<CoqDatasetAugmentedFile>((_, reject) => {
                setTimeout(() => {
                    reject(
                        new CoqLspTimeoutError(
                            `checkProofs timed out after ${fileAugmentationTimeoutMillis} milliseconds`
                        )
                    );
                }, fileAugmentationTimeoutMillis);
            });

            return Promise.race([
                this.augmentFileWithValidSamplesUnsafe(
                    sourceDirPath,
                    unaugmentedFile,
                    fileTypeCheckingTimeoutMillis
                ),
                timeoutPromise,
            ]);
        });
    }

    private buildAuxFileUri(
        sourceDirPath: string,
        initialFileName: string, 
        unique: boolean = true
    ): Uri {
        const defaultAuxFileName = `${initialFileName}_augmentation_validation_cp_aux.v`;
        let auxFilePath = path.join(sourceDirPath, defaultAuxFileName);
        if (unique && existsSync(auxFilePath)) {
            const randomSuffix = Math.floor(Math.random() * 1000000);
            auxFilePath = auxFilePath.replace(
                /\_cp_aux.v$/,
                `_${randomSuffix}_cp_aux.v`
            );
        }

        return Uri.fromPath(auxFilePath);
    }

    private async augmentFileWithValidSamplesUnsafe(
        sourceDirPath: string,
        unaugmentedFile: CoqDatasetUnaugmentedFile,
        fileTypeCheckingTimeoutMillis: number
    ): Promise<CoqDatasetAugmentedFile> {
        const auxFileUri = this.buildAuxFileUri(
            sourceDirPath,
            unaugmentedFile.itemName
        );
        const fileContent = unaugmentedFile.initialFileContent;

        const coqDatasetFileItem: CoqDatasetAugmentedFile = {
            ...unaugmentedFile,
            augmentedTheorems: [],
            stats: emptyDatasetStats(),
            type: "file",
        };

        if (unaugmentedFile.theoremsWithProofTrees.length === 0) {
            return coqDatasetFileItem;
        }

        // Content in range [file_start, first_theorem_start]
        const fileStartPos = { line: 0, character: 0 };
        const fileEndPos = {
            line: fileContent.length - 1,
            character: fileContent[fileContent.length - 1].length,
        };
    
        const firstTheoremStart =
            unaugmentedFile.theoremsWithProofTrees[0].parsedTheorem.statement_range.start;
    
        let validationFileCurPrefix = getTextInRange(
            fileStartPos,
            firstTheoremStart,
            fileContent,
            true
        );
        writeFileSync(auxFileUri.fsPath, validationFileCurPrefix);
        const progress = getProgressBar("Validating new proofs for", unaugmentedFile.itemName, unaugmentedFile.theoremsWithProofTrees.length);

        try { 
            await this.coqLspClient.openTextDocument(
                auxFileUri,
                undefined,
                fileTypeCheckingTimeoutMillis
            );
            let auxFileVersion = 1;

            for (let i = 0; i < unaugmentedFile.theoremsWithProofTrees.length; i++) {
                // 1. Validate samples for current theorem
                const theorem = unaugmentedFile.theoremsWithProofTrees[i];
                
                // 1.1. There is no proof-tree, adding empty stats
                if (theorem.proofTreeBuildResult.err) {
                    const theoremItem: CoqDatasetTheoremItem =
                        createTheoremWithStats(
                            theorem.parsedTheorem,
                            theorem.sourceFilePath,
                            theorem.proofTreeBuildResult
                        );
                    coqDatasetFileItem.augmentedTheorems.push(theoremItem);
                    coqDatasetFileItem.stats = accumulateTheoremStats(
                        coqDatasetFileItem.stats,
                        theoremItem.stats
                    );
                } else {
                    // 1.2. There is a proof-tree, augmenting it

                    const samples = augmentTreeToSamples(
                        theorem.proofTreeBuildResult.val,
                        theorem.parsedTheorem.name
                    );
                    const validationResults: Map<number, TheoremValidationResult> = new Map();

                    for (const [index, theorem] of samples) {
                        auxFileVersion += 1;
                        const appendedText = `\n\n${theoremDatasetSampleToString(theorem)}`;
                        appendFileSync(auxFileUri.fsPath, appendedText);
        
                        try { 
                            const diagnosticMessage =
                                await this.coqLspClient.updateTextDocument(
                                    validationFileCurPrefix,
                                    appendedText,
                                    auxFileUri,
                                    auxFileVersion,
                                    fileTypeCheckingTimeoutMillis
                                );
            
                            validationResults.set(index, {
                                theorem,
                                isValid: diagnosticMessage === undefined,
                                diagnostic: diagnosticMessage,
                            });
            
                            // 1.3.1. Reset file to the previous state
                            writeFileSync(auxFileUri.fsPath, validationFileCurPrefix);
                            auxFileVersion += 1;
                            await this.coqLspClient.updateTextDocument(
                                validationFileCurPrefix,
                                "",
                                auxFileUri,
                                auxFileVersion,
                                fileTypeCheckingTimeoutMillis
                            );
                        } catch (error) {
                            if (error instanceof CoqLspError) {
                                this.eventLogger?.log(
                                    "timeout",
                                    `Timeout while checking theorem ${theorem.theoremStatement}`
                                );

                                validationResults.set(index, {
                                    theorem,
                                    isValid: false,
                                    diagnostic: "Timeout reached",
                                });
                            }
                        }
                    }

                    // 1.4 Use the validation results to build the augmented theorem
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
                        this.eventLogger?.log(
                            "node-augmentation-result",
                            `Node ${nodeId} augmentation result: ${res}`
                        );
                    });

                    const coqAugmentedTheorem: CoqAugmentedTheoremItem = {
                        samples: nodeAugmentationRes,
                        proofTree: theorem.proofTreeBuildResult.val,
                    };

                    const augmentedTheoremItem: CoqDatasetTheoremItem =
                        createTheoremWithStats(
                            theorem.parsedTheorem,
                            theorem.sourceFilePath,
                            Ok(coqAugmentedTheorem)
                        );

                    coqDatasetFileItem.augmentedTheorems.push(augmentedTheoremItem);
                    coqDatasetFileItem.stats = accumulateTheoremStats(
                        coqDatasetFileItem.stats,
                        augmentedTheoremItem.stats
                    );
                }

                // 2. Append new content (in-between with the next theorem) to the file
                const theoremStart = theorem.parsedTheorem.statement_range.start;
                const theoremEnd = theorem.parsedTheorem.proof.end_pos.end;
        
                const theoremContent = getTextInRange(
                    theoremStart,
                    theoremEnd,
                    fileContent,
                    true
                );
                const nextPoint =
                    i + 1 < unaugmentedFile.theoremsWithProofTrees.length
                        ? unaugmentedFile.theoremsWithProofTrees[i + 1].parsedTheorem
                                .statement_range.start
                        : fileEndPos;
        
                const contentAfterTheorem = getTextInRange(
                    theoremEnd,
                    nextPoint,
                    fileContent,
                    true
                );
        
                validationFileCurPrefix +=
                    theoremContent + contentAfterTheorem;
                writeFileSync(auxFileUri.fsPath, validationFileCurPrefix);

                auxFileVersion += 1;
                await this.coqLspClient.updateTextDocument(
                    validationFileCurPrefix,
                    "",
                    auxFileUri,
                    auxFileVersion,
                    fileTypeCheckingTimeoutMillis
                );

                progress.update();
            }
        } finally {
            await this.coqLspClient.closeTextDocument(auxFileUri);
            unlinkSync(auxFileUri.fsPath);
        }

        return coqDatasetFileItem;
    }

    dispose(): void {
        this.coqLspClient.dispose();
    }
}
