import { Mutex } from "async-mutex";
import { appendFileSync, existsSync, unlinkSync, writeFileSync } from "fs";
import * as path from "path";
import { Position } from "vscode-languageclient";

import { CoqLspClient } from "../coqLsp/coqLspClient";
import { CoqLspTimeoutError } from "../coqLsp/coqLspTypes";

import { TheoremDatasetSample } from "../coqDatasetRepresentation/coqDatasetModels";
import { Uri } from "../utils/uri";

import { theoremDatasetSampleToString } from "./proofBuilder";

export interface TheoremValidationResult {
    theorem: TheoremDatasetSample;
    isValid: boolean;
    diagnostic?: string;
}

export interface CoqTheoremValidatorInterface {
    /**
     * To check that the pack of generated theorems is valid and the make sense.
     */
    validateTheorems(
        sourceDirPath: string,
        sourceFileContentPrefix: string[],
        prefixEndPosition: Position,
        theoremsToCheck: Map<number, TheoremDatasetSample>
    ): Promise<Map<number, TheoremValidationResult>>;

    dispose(): void;
}

export class CoqTheoremValidator implements CoqTheoremValidatorInterface {
    private mutex: Mutex = new Mutex();

    constructor(private coqLspClient: CoqLspClient) {}

    async validateTheorems(
        sourceDirPath: string,
        sourceFileContentPrefix: string[],
        prefixEndPosition: Position,
        theoremsToCheck: Map<number, TheoremDatasetSample>,
        coqLspTimeoutMillis: number = 15000
    ): Promise<Map<number, TheoremValidationResult>> {
        return await this.mutex.runExclusive(async () => {
            const timeoutPromise = new Promise<Map<number, TheoremValidationResult>>(
                (_, reject) => {
                    setTimeout(() => {
                        reject(
                            new CoqLspTimeoutError(
                                `checkProofs timed out after ${coqLspTimeoutMillis} milliseconds`
                            )
                        );
                    }, coqLspTimeoutMillis);
                }
            );

            return Promise.race([
                this.validateTheoremsUnsafe(
                    sourceDirPath,
                    sourceFileContentPrefix,
                    prefixEndPosition,
                    theoremsToCheck
                ),
                timeoutPromise,
            ]);
        });
    }

    private buildAuxFileUri(
        sourceDirPath: string,
        holePosition: Position,
        unique: boolean = true
    ): Uri {
        const holeIdentifier = `${holePosition.line}_${holePosition.character}`;
        const defaultAuxFileName = `hole_${holeIdentifier}_cp_aux.v`;
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

    private async validateTheoremsUnsafe(
        sourceDirPath: string,
        sourceFileContentPrefix: string[],
        prefixEndPosition: Position,
        theoremsToCheck: Map<number, TheoremDatasetSample>
    ): Promise<Map<number, TheoremValidationResult>> {
        const auxFileUri = this.buildAuxFileUri(
            sourceDirPath,
            prefixEndPosition
        );
        const sourceFileContent = sourceFileContentPrefix.join("\n");
        writeFileSync(auxFileUri.fsPath, sourceFileContent);

        const results: Map<number, TheoremValidationResult> = new Map();
        try {
            await this.coqLspClient.openTextDocument(auxFileUri);
            let auxFileVersion = 1;

            for (const [index, theorem] of theoremsToCheck) {
                auxFileVersion += 1;
                const appendedText = `\n\n${theoremDatasetSampleToString(theorem)}`;
                appendFileSync(auxFileUri.fsPath, appendedText);

                const diagnosticMessage =
                    await this.coqLspClient.updateTextDocument(
                        sourceFileContentPrefix,
                        appendedText,
                        auxFileUri,
                        auxFileVersion
                    );

                results.set(index, {
                    theorem,
                    isValid: diagnosticMessage === undefined,
                    diagnostic: diagnosticMessage,
                });

                writeFileSync(auxFileUri.fsPath, sourceFileContent);

                auxFileVersion += 1;
                await this.coqLspClient.updateTextDocument(
                    sourceFileContentPrefix,
                    "",
                    auxFileUri,
                    auxFileVersion
                );
            }
        } finally {
            await this.coqLspClient.closeTextDocument(auxFileUri);
            unlinkSync(auxFileUri.fsPath);
        }

        return results;
    }

    dispose(): void {
        this.coqLspClient.dispose();
    }
}
