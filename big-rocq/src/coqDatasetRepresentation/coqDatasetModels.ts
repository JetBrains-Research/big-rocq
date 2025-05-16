import { Result } from "ts-results";

import { Theorem } from "../coqParser/parsedTypes";
import { CoqProofTree } from "../coqProofTree/coqProofTree";

// TODO: Separate visualisation from core dataset models
export interface TheoremDatasetSample {
    namedTheoremStatement: string;
    statement: string;
    conclusion: string;
    hypotheses: string;
    proof: string;
    proofLength: number;
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
    stats: CoqTheoremStats;
}

export interface CoqDatasetUnvalidatedTheorem {
    parsedTheorem: Theorem;
    sourceFilePath: string;
    proofTreeBuildResult: Result<CoqProofTree, Error>;
}

export interface CoqDatasetAugmentedFile {
    filePath: string;
    itemName: string;
    initialFileContent: string[];
    augmentedTheorems: CoqDatasetTheoremItem[];
    stats: CoqDatasetStats;
    type: "file";
}

export interface CoqDatasetUnaugmentedFile {
    filePath: string;
    itemName: string;
    initialFileContent: string[];
    theoremsWithProofTrees: CoqDatasetUnvalidatedTheorem[];
}

export interface CoqDatasetFolder {
    dirPath: string;
    itemName: string;
    stats: CoqDatasetStats;
    dirItems: CoqDatasetDirItem[];
    type: "dir";
}

export interface CoqDatasetStats extends CoqTheoremStats {
    // Lines of code before and after augmentation
    locChangeAfterAugmentation: [number, number];
    processingTimeMillis: number;
}

export interface CoqTheoremStats {
    augmentedNodesRatio: Ratio;
    proofTreeBuildRatio: Ratio;
    augmentedProofLengthCounts: Map<number, number>;
}

export type Ratio = [number, number];

export type CoqDatasetDirItem = CoqDatasetAugmentedFile | CoqDatasetFolder;

export type CoqDataset = CoqDatasetDirItem;
