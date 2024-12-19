import { Result } from "ts-results";

import { PpMode } from "../../utils/proofStatePrinters";
import { CoqDatasetTheoremItem } from "../coqDatasetModels";

import {
    CompactSerializedCoqProofTree,
    compactSerializeCoqProofTree,
} from "./compactSerialization/compactProofTree";

export type CompactSerializedTheoremAugmentationResult = Result<
    CompactSerializedCoqProofTree,
    Error
>;

export interface CompactSerializedCoqDatasetTheoremItem {
    sourceFile: string;
    augmentedTheorem: CompactSerializedTheoremAugmentationResult;
}

export interface CompactSerializedCoqDatasetAugmentedFile {
    filePath: string;
    augmentedTheorems: CompactSerializedCoqDatasetTheoremItem[];
}

export interface CompactSerializedCoqDataset {
    dataset: CompactSerializedCoqDatasetAugmentedFile[];
}

export function compactSerializeCoqTheorem(
    theorem: CoqDatasetTheoremItem,
    ppType: PpMode
): CompactSerializedTheoremAugmentationResult {
    return theorem.theoremAugmentationRes.map((item) =>
        compactSerializeCoqProofTree(
            item.proofTree,
            ppType,
            theorem.parsedTheorem.name
        )
    );
}
