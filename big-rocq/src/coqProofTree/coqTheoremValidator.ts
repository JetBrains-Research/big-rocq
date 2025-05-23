import * as assert from "assert";
import { Mutex } from "async-mutex";
import { appendFileSync, existsSync, unlinkSync, writeFileSync } from "fs";
import * as path from "path";
import { Err, Ok } from "ts-results";

import { createCoqLspClient } from "../coqLsp/coqLspBuilders";
import { CoqLspClient } from "../coqLsp/coqLspClient";
import {
    CoqLspError,
    CoqLspTimeoutError,
    Hyp,
    PpString,
} from "../coqLsp/coqLspTypes";

import {
    CoqAugmentedTheoremItem,
    CoqDatasetAugmentedFile,
    CoqDatasetTheoremItem,
    CoqDatasetUnaugmentedFile,
    CoqDatasetUnvalidatedTheorem,
    NodeAugmentationResult,
    TheoremDatasetSample,
} from "../coqDatasetRepresentation/coqDatasetModels";
import {
    accumulateTheoremStats,
    createTheoremWithStats,
    emptyDatasetStats,
} from "../coqDatasetRepresentation/statisticsCalculation";
import { RunParams } from "../core/utilityRunParams";
import { EventLogger } from "../logging/eventLogger";
import { getProgressBar } from "../logging/progressBar";
import { getTextInRange } from "../utils/documentUtils";
import { Uri } from "../utils/uri";

import {
    augmentTreeToSamples,
    theoremDatasetSampleToString,
} from "./proofBuilder";

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
        public readonly abortController: AbortController,
        public readonly runArgs: RunParams,
        public readonly eventLogger?: EventLogger
    ) {}

    async augmentFileWithValidSamples(
        sourceDirPath: string,
        unaugmentedFile: CoqDatasetUnaugmentedFile,
        fileAugmentationTimeoutMillis: number = 150000,
        fileTypeCheckingTimeoutMillis: number = 150000
    ): Promise<CoqDatasetAugmentedFile> {
        return await this.mutex.runExclusive(async () => {
            const coqLspClient = await createCoqLspClient(
                this.runArgs.workspaceRootPath,
                this.runArgs.coqLspServerPath,
                undefined,
                this.eventLogger,
                this.abortController
            );

            const timeoutPromise = new Promise<CoqDatasetAugmentedFile>(
                (_, reject) => {
                    setTimeout(() => {
                        coqLspClient.dispose();

                        reject(
                            new CoqLspTimeoutError(
                                `checkProofs timed out after ${fileAugmentationTimeoutMillis} milliseconds`
                            )
                        );
                    }, fileAugmentationTimeoutMillis);
                }
            );

            return Promise.race([
                this.augmentFileWithValidSamplesUnsafe(
                    sourceDirPath,
                    unaugmentedFile,
                    fileTypeCheckingTimeoutMillis,
                    coqLspClient
                ),
                timeoutPromise,
            ]);
        });
    }

    private async augmentFileWithValidSamplesUnsafe(
        sourceDirPath: string,
        unaugmentedFile: CoqDatasetUnaugmentedFile,
        fileTypeCheckingTimeoutMillis: number,
        coqLspClient: CoqLspClient
    ): Promise<CoqDatasetAugmentedFile> {
        const auxFileUri = this.buildAuxFileUri(
            sourceDirPath,
            unaugmentedFile.itemName
        );
        const fileContent = unaugmentedFile.initialFileContent;
        const totalThrCount = unaugmentedFile.theoremsWithProofTrees.length;

        let coqDatasetFileItem: CoqDatasetAugmentedFile = {
            ...unaugmentedFile,
            augmentedTheorems: [],
            stats: emptyDatasetStats(),
            type: "file",
        };

        if (totalThrCount === 0) {
            return coqDatasetFileItem;
        }

        const fileEndPos = this.getFileEndPos(fileContent);

        // Content in range [file_start, first_theorem_start]
        let validationFileCurPrefix = this.getTextFromBeginningToFirstTheorem(
            unaugmentedFile,
            fileContent
        );

        writeFileSync(auxFileUri.fsPath, validationFileCurPrefix);
        const progress = getProgressBar(
            "Validating new proofs for",
            unaugmentedFile.itemName,
            totalThrCount
        );

        try {
            await coqLspClient.openTextDocument(
                auxFileUri,
                undefined,
                fileTypeCheckingTimeoutMillis
            );
            let auxFileVersion = 1;

            for (let i = 0; i < totalThrCount; i++) {
                // 1. Validate samples for current theorem
                const theorem = unaugmentedFile.theoremsWithProofTrees[i];

                let globalContext: HypothesesDict;
                [globalContext, auxFileVersion] =
                    await this.getCurrentGlobalHyps(
                        auxFileUri,
                        auxFileVersion,
                        validationFileCurPrefix,
                        fileTypeCheckingTimeoutMillis,
                        coqLspClient
                    );

                // 1.1. There is no proof-tree, adding empty stats
                if (theorem.proofTreeBuildResult.err) {
                    coqDatasetFileItem =
                        this.augmentFileWithUnsuccessfulTheorem(
                            coqDatasetFileItem,
                            theorem
                        );
                } else {
                    // 1.2. There is a proof-tree, augmenting it
                    const samples = augmentTreeToSamples(
                        theorem.proofTreeBuildResult.val,
                        theorem.parsedTheorem.name,
                        globalContext
                    );

                    let validationResults: ValidatedAugmentationSamples;
                    [validationResults, auxFileVersion] =
                        await this.validateSamplesForTheorem(
                            samples,
                            auxFileUri,
                            auxFileVersion,
                            validationFileCurPrefix,
                            fileTypeCheckingTimeoutMillis,
                            coqLspClient
                        );

                    // 1.4 Use the validation results to build the augmented theorem
                    coqDatasetFileItem = this.augmentFileWithNewTheorem(
                        coqDatasetFileItem,
                        theorem,
                        samples,
                        validationResults
                    );
                }

                // 2. Append new content (in-between with the next theorem) to the file
                const appendedContext = this.getAppendedContentAfterTheorem(
                    i,
                    theorem,
                    unaugmentedFile,
                    fileContent,
                    totalThrCount,
                    fileEndPos
                );

                validationFileCurPrefix += appendedContext;
                writeFileSync(auxFileUri.fsPath, validationFileCurPrefix);

                auxFileVersion += 1;
                await coqLspClient.updateTextDocument(
                    validationFileCurPrefix,
                    "",
                    auxFileUri,
                    auxFileVersion,
                    fileTypeCheckingTimeoutMillis
                );

                progress.update();
            }
        } finally {
            await coqLspClient.closeTextDocument(auxFileUri);
            unlinkSync(auxFileUri.fsPath);
            coqLspClient.dispose();
        }

        return coqDatasetFileItem;
    }

    private async getCurrentGlobalHyps(
        auxFileUri: Uri,
        auxFileVersion: number,
        validationFileCurPrefix: string,
        fileTypeCheckingTimeoutMillis: number,
        coqLspClient: CoqLspClient
    ): Promise<[HypothesesDict, FileVersion]> {
        auxFileVersion += 1;

        const appendedText = `\n\n${CoqTheoremValidator.CHECK_GLOBAL_STATE_HACK_THEOREM}`;
        const fullPrefixWithAuxTheorem = (
            validationFileCurPrefix + appendedText
        ).split("\n");
        const auxTheoremEndPos = this.getFileEndPos(fullPrefixWithAuxTheorem);
        appendFileSync(auxFileUri.fsPath, appendedText);
        await coqLspClient.updateTextDocument(
            validationFileCurPrefix,
            appendedText,
            auxFileUri,
            auxFileVersion,
            fileTypeCheckingTimeoutMillis
        );

        const goalsAtAuxTheorem = await coqLspClient.getGoalsAtPoint(
            auxTheoremEndPos,
            auxFileUri,
            auxFileVersion
        );

        assert(goalsAtAuxTheorem.ok);
        assert(goalsAtAuxTheorem.val.goals.length === 1);

        const hyps = this.flattenHyps(goalsAtAuxTheorem.val.goals[0].hyps);

        // Reset file to the previous state
        writeFileSync(auxFileUri.fsPath, validationFileCurPrefix);
        auxFileVersion += 1;

        await coqLspClient.updateTextDocument(
            validationFileCurPrefix,
            "",
            auxFileUri,
            auxFileVersion,
            fileTypeCheckingTimeoutMillis
        );

        return [hyps, auxFileVersion];
    }

    private static readonly CHECK_GLOBAL_STATE_HACK_THEOREM =
        "Theorem impossible_theorem_name : 52 = 52.";

    private flattenHyps(hyps: Hyp<PpString>[]) {
        const hypsDict = new Map<string, string>();
        hyps.forEach((hyp) => {
            hyp.names.forEach((name) => {
                hypsDict.set(name as string, hyp.ty as string);
            });
        });

        return hypsDict;
    }

    private async validateSamplesForTheorem(
        samples: Map<number, TheoremDatasetSample>,
        auxFileUri: Uri,
        auxFileVersion: number,
        validationFileCurPrefix: string,
        fileTypeCheckingTimeoutMillis: number,
        coqLspClient: CoqLspClient
    ): Promise<[ValidatedAugmentationSamples, FileVersion]> {
        const validationResults: Map<number, TheoremValidationResult> =
            new Map();

        for (const [index, theorem] of samples) {
            auxFileVersion += 1;
            const appendedText = `\n\n${theoremDatasetSampleToString(theorem)}`;
            appendFileSync(auxFileUri.fsPath, appendedText);

            try {
                const diagnosticMessage = await coqLspClient.updateTextDocument(
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
                await coqLspClient.updateTextDocument(
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
                        `Timeout while checking theorem ${theorem.namedTheoremStatement}`
                    );

                    validationResults.set(index, {
                        theorem,
                        isValid: false,
                        diagnostic: "Timeout reached",
                    });
                }
            }
        }

        return [validationResults, auxFileVersion];
    }

    private getFileEndPos(fileContent: string[]): {
        line: number;
        character: number;
    } {
        return {
            line: fileContent.length - 1,
            character: fileContent[fileContent.length - 1].length,
        };
    }

    private getAppendedContentAfterTheorem(
        index: number,
        theorem: CoqDatasetUnvalidatedTheorem,
        unaugmentedFile: CoqDatasetUnaugmentedFile,
        fileContent: string[],
        totalThrCount: number,
        fileEndPos: { line: number; character: number }
    ): string {
        const theoremStart = theorem.parsedTheorem.statement_range.start;
        const theoremEnd = theorem.parsedTheorem.proof.end_pos.end;

        const theoremContent = getTextInRange(
            theoremStart,
            theoremEnd,
            fileContent,
            true
        );
        const nextPoint =
            index + 1 < totalThrCount
                ? unaugmentedFile.theoremsWithProofTrees[index + 1]
                      .parsedTheorem.statement_range.start
                : fileEndPos;

        const contentAfterTheorem = getTextInRange(
            theoremEnd,
            nextPoint,
            fileContent,
            true
        );

        return theoremContent + contentAfterTheorem;
    }

    private getTextFromBeginningToFirstTheorem(
        unaugmentedFile: CoqDatasetUnaugmentedFile,
        fileContent: string[]
    ): string {
        const fileStartPos = { line: 0, character: 0 };
        const firstTheoremStart =
            unaugmentedFile.theoremsWithProofTrees[0].parsedTheorem
                .statement_range.start;

        return getTextInRange(
            fileStartPos,
            firstTheoremStart,
            fileContent,
            true
        );
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

    private augmentFileWithUnsuccessfulTheorem(
        coqDatasetFileItem: CoqDatasetAugmentedFile,
        theorem: CoqDatasetUnvalidatedTheorem
    ): CoqDatasetAugmentedFile {
        assert(theorem.proofTreeBuildResult.err);

        const theoremItem: CoqDatasetTheoremItem = createTheoremWithStats(
            theorem.parsedTheorem,
            theorem.sourceFilePath,
            theorem.proofTreeBuildResult
        );
        coqDatasetFileItem.augmentedTheorems.push(theoremItem);
        coqDatasetFileItem.stats = accumulateTheoremStats(
            coqDatasetFileItem.stats,
            theoremItem.stats
        );

        return coqDatasetFileItem;
    }

    private augmentFileWithNewTheorem(
        coqDatasetFileItem: CoqDatasetAugmentedFile,
        theorem: CoqDatasetUnvalidatedTheorem,
        unvalidatedSamples: Map<number, TheoremDatasetSample>,
        validationResults: Map<number, TheoremValidationResult>
    ): CoqDatasetAugmentedFile {
        assert(theorem.proofTreeBuildResult.ok);

        const nodeAugmentationRes = new Map<number, NodeAugmentationResult>();

        unvalidatedSamples.forEach((sample, nodeId) => {
            const validationRes = validationResults.get(nodeId);
            if (validationRes === undefined) {
                throw new Error(`No validation result for node ${nodeId}`);
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

        return coqDatasetFileItem;
    }

    dispose(): void {}
}

type ValidatedAugmentationSamples = Map<number, TheoremValidationResult>;
type FileVersion = number;
export type HypothesesDict = Map<string, string>;
