import { writeFileSync } from "fs";
import * as path from "path";

import { ensureDirExists } from "../utils/documentUtils";

import { CoqDatasetAugmentedFile } from "./coqDatasetModels";
import { CompactSerializedAugmentedSamples } from "./serialization/augmentedSamplesSerialization";

// TODO: Refactor to params
const coqAugmentedSamplesDir = "tempDatasetView/augmentedSamples";

export function storeCoqAugmentedSamples(
    rootPath: string,
    coqDatasetFile: CoqDatasetAugmentedFile
) {
    ensureDirExists(coqAugmentedSamplesDir);
    const relativeFromRoot = path.relative(rootPath, coqDatasetFile.filePath);

    const fileSubdir = path.dirname(relativeFromRoot);
    const fileName = path.parse(coqDatasetFile.filePath).name;
    const fileDir = path.join(coqAugmentedSamplesDir, fileSubdir);
    ensureDirExists(fileDir);

    const storedDatasetPath = path.join(fileDir, `${fileName}_statements.json`);
    const storedSamplesContent = augmentedSamplesAsJson(coqDatasetFile);

    writeFileSync(storedDatasetPath, storedSamplesContent);
}

function augmentedSamplesAsJson(
    coqDatasetFile: CoqDatasetAugmentedFile
): string {
    const augmentedFileSamples: CompactSerializedAugmentedSamples = {
        filePath: coqDatasetFile.filePath,
        fileSamples: [],
    };

    coqDatasetFile.augmentedTheorems.forEach((theorem) => {
        if (theorem.proofTreeBuildResult.ok) {
            const samples = theorem.proofTreeBuildResult.val.samples;
            samples.delete(0);

            Array.from(samples.entries()).forEach(([_, result]) => {
                if (result.ok) {
                    augmentedFileSamples.fileSamples.push({
                        statement: result.val.statement,
                        conclusion: result.val.conclusion,
                        hypotheses: result.val.hypotheses,
                        proofString: result.val.proof,
                    });
                }
            });
        }
    });

    return JSON.stringify(augmentedFileSamples);
}
