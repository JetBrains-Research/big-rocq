import { Result } from "ts-results";

import { Theorem } from "../coqParser/parsedTypes";
import { CoqProofTree } from "../coqProofTree/coqProofTree";

// TODO: Separate visualisation from core dataset models
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
    sourceFilePath: string;
    proofTreeBuildResult: ProofTreeBuildResult;
    stats: CoqDatasetStats;
}

export interface CoqDatasetAugmentedFile {
    filePath: string;
    itemName: string;
    initialFileContent: string[];
    augmentedTheorems: CoqDatasetTheoremItem[];
    stats: CoqDatasetStats;
    type: "file";
}

export interface CoqDatasetFolder {
    dirPath: string;
    itemName: string;
    stats: CoqDatasetStats;
    dirItems: CoqDatasetDirItem[];
    type: "dir";
}

export interface CoqDatasetStats {
    augmentedNodesRatio: Ratio;
    proofTreeBuildRatio: Ratio;
}

export type Ratio = [number, number];

export type CoqDatasetDirItem = CoqDatasetAugmentedFile | CoqDatasetFolder;

export type CoqDataset = CoqDatasetDirItem;
