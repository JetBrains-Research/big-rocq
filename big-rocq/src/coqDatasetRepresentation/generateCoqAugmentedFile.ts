import { writeFileSync } from "fs";
import * as path from "path";

import { theoremDatasetSampleToString } from "../coqProofTree/proofBuilder";
import { ensureDirExists, getTextInRange } from "../utils/documentUtils";

import {
    CoqDatasetAugmentedFile,
    CoqDatasetTheoremItem,
    TheoremDatasetSample,
} from "./coqDatasetModels";

// TODO: Refactor to params
const coqAugmentedFilesDir = "tempDatasetView/augmentedFiles";

/**
 * @param rootPath The relative root path of the project being processed
 * @param coqDatasetFile The processed data from which the augmented file will be generated
 *
 * @returns Returns the length of the generated file for statistics calculation
 */
export function generateCoqAugmentedFile(
    rootPath: string,
    coqDatasetFile: CoqDatasetAugmentedFile,
    debug: boolean = false
): number {
    ensureDirExists(coqAugmentedFilesDir);
    const relativeFromRoot = path.relative(rootPath, coqDatasetFile.filePath);

    const fileSubdir = path.dirname(relativeFromRoot);
    const fileName = path.parse(coqDatasetFile.filePath).name;
    const fileDir = path.join(coqAugmentedFilesDir, fileSubdir);
    ensureDirExists(fileDir);

    const augmentedFilePath = path.join(fileDir, `${fileName}_augmented.v`);
    const augmentedFileContent = createCoqAugmentedFile(coqDatasetFile, debug);

    writeFileSync(augmentedFilePath, augmentedFileContent);
    return augmentedFileContent.split("\n").length;
}

function createCoqAugmentedFile(
    coqDatasetFile: CoqDatasetAugmentedFile,
    debug: boolean
): string {
    const fileContent = coqDatasetFile.initialFileContent;

    // Assert that exists at least one theorem
    if (coqDatasetFile.augmentedTheorems.length === 0) {
        return fileContent.join("\n");
    }

    // Content in range [file_start, first_theorem_start]
    const fileStartPos = { line: 0, character: 0 };
    const fileEndPos = {
        line: fileContent.length - 1,
        character: fileContent[fileContent.length - 1].length,
    };

    const firstTheoremStart =
        coqDatasetFile.augmentedTheorems[0].parsedTheorem.statement_range.start;

    let collectedContent = getTextInRange(
        fileStartPos,
        firstTheoremStart,
        fileContent,
        true
    );

    for (let i = 0; i < coqDatasetFile.augmentedTheorems.length; i++) {
        const theorem = coqDatasetFile.augmentedTheorems[i];
        const theoremStart = theorem.parsedTheorem.statement_range.start;
        const theoremEnd = theorem.parsedTheorem.proof.end_pos.end;

        const theoremContent = getTextInRange(
            theoremStart,
            theoremEnd,
            fileContent,
            true
        );
        const nextPoint =
            i + 1 < coqDatasetFile.augmentedTheorems.length
                ? coqDatasetFile.augmentedTheorems[i + 1].parsedTheorem
                      .statement_range.start
                : fileEndPos;

        const contentAfterTheorem = getTextInRange(
            theoremEnd,
            nextPoint,
            fileContent,
            true
        );
        const generatedSamples = aggregateGeneratedSamples(theorem, debug);

        collectedContent +=
            theoremContent + generatedSamples + contentAfterTheorem;
    }

    return collectedContent;
}

function aggregateGeneratedSamples(
    theorem: CoqDatasetTheoremItem,
    debug: boolean
): string {
    if (theorem.proofTreeBuildResult.err) {
        return debug
            ? "\n\n(** Error: Proof tree build failed, no samples available. **)"
            : "";
    }

    const samples = theorem.proofTreeBuildResult.val.samples;
    // Remove augmented sample that represents the augmented
    // sample with zero tactics removed from the proof
    samples.delete(0);

    if (samples.size === 0) {
        return debug ? "\n\n(** No samples available. **)" : "";
    }

    let result = "\n\n";
    if (debug) {
        result +=
            "\n\n(** Successfully built proof tree. Successfully augmented " +
            theorem.stats.augmentedNodesRatio[0] +
            "/" +
            theorem.stats.augmentedNodesRatio[1] +
            " non-trivial nodes (after initial) **)\n\n";
    }

    const sortedSamples = Array.from(samples.entries())
        .filter(([_, result]) => result.ok) // Only include successful results
        .sort(([id1], [id2]) => id1 - id2) // Sort by NodeId
        .map(([_, result]) =>
            theoremDatasetSampleToString(result.val as TheoremDatasetSample)
        );

    result += sortedSamples.join("\n\n");

    if (debug) {
        return result + "\n\n(** End of samples **)";
    }

    return result;
}
