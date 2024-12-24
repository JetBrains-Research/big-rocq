import { Result } from "ts-results";

import { Theorem } from "../coqParser/parsedTypes";
import { CoqProofTree } from "../coqProofTree/coqProofTree";

export interface TheoremDatasetSample {
    theoremStatement: string;
    proof: string;
}

export type NodeAugmentationResult = Result<TheoremDatasetSample, Error>;
export type NodeId = number;

export interface CoqAugmentedTheoremItem {
    samples: Map<NodeId, NodeAugmentationResult>; 
    proofTree: CoqProofTree;
}

export type ProofTreeBuildResult = Result<CoqAugmentedTheoremItem, Error>;

export interface CoqDatasetTheoremItem {
    parsedTheorem: Theorem;
    sourceFile: string;
    proofTreeBuildResult: ProofTreeBuildResult;
}

export interface CoqDatasetAugmentedFile {
    filePath: string;
    augmentedTheorems: CoqDatasetTheoremItem[];
}

export type CoqDataset = CoqDatasetAugmentedFile[];
