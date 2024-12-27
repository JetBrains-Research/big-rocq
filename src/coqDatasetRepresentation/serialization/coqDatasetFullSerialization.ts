import { Result } from "ts-results";

import {
    NodeAugmentationResult,
    NodeId,
} from "../coqDatasetModels";

import {
    SerializedCoqProofTree,
} from "./serializationUtils/coqProofTree";
import {
    SerializedTheorem,
} from "./serializationUtils/theoremData";

export interface SerializedCoqAugmentedTheoremItem {
    samples: Map<NodeId, NodeAugmentationResult>;
    proofTree: SerializedCoqProofTree;
}

export type SerializedTheoremAugmentationResult = Result<
    SerializedCoqAugmentedTheoremItem,
    Error
>;

export interface SerializedCoqDatasetTheoremItem {
    parsedTheorem: SerializedTheorem;
    sourceFile: string;
    augmentedTheorem: SerializedTheoremAugmentationResult;
}

export interface SerializedCoqDatasetAugmentedFile {
    filePath: string;
    augmentedTheorems: SerializedCoqDatasetTheoremItem[];
}

export interface SerializedCoqDataset {
    dataset: SerializedCoqDatasetAugmentedFile[];
}

// TODO: Fix serialization

// function serializeCoqAugmentedTheoremItem(
//     item: CoqAugmentedTheoremItem
// ): SerializedCoqAugmentedTheoremItem {
//     return {
//         samples: item.samples,
//         proofTree: serializeCoqProofTree(item.proofTree),
//     };
// }

// function serializeCoqDatasetTheoremItem(
//     item: CoqDatasetTheoremItem,
//     theoremIndex: number
// ): SerializedCoqDatasetTheoremItem {
//     return {
//         parsedTheorem: serializeTheoremData(
//             new TheoremData(item.parsedTheorem, theoremIndex)
//         ),
//         sourceFile: item.sourceFile,
//         augmentedTheorem: item.proofTreeBuildResult.map(
//             serializeCoqAugmentedTheoremItem
//         ),
//     };
// }

// function serializeCoqDatasetAugmentedFile(
//     file: CoqDatasetAugmentedFile
// ): SerializedCoqDatasetAugmentedFile {
//     return {
//         filePath: file.filePath,
//         augmentedTheorems: file.augmentedTheorems.map(
//             serializeCoqDatasetTheoremItem
//         ),
//     };
// }

// export function serializeCoqDataset(dataset: CoqDataset): SerializedCoqDataset {
//     return {
//         dataset: dataset.map(serializeCoqDatasetAugmentedFile),
//     };
// }
