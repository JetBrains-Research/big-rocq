import { Result } from "ts-results";

import { Theorem } from "../coqParser/parsedTypes";
import { CoqProofTree } from "../coqProofTree/coqProofTree";

export interface TheoremDatasetSample {
    theoremStatement: string;
    proof: string;
}

export interface CoqAugmentedTheoremItem {
    samples: TheoremDatasetSample[];
    proofTree: CoqProofTree;
}

export type TheoremAugmentationResult = Result<CoqAugmentedTheoremItem, Error>;

export interface CoqDatasetTheoremItem {
    parsedTheorem: Theorem;
    sourceFile: string;
    theoremAugmentationRes: TheoremAugmentationResult;
}

export interface CoqDatasetAugmentedFile {
    filePath: string;
    augmentedTheorems: CoqDatasetTheoremItem[];
}

export type CoqDataset = CoqDatasetAugmentedFile[];
