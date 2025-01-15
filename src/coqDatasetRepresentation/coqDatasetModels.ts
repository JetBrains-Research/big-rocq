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
    sourceFile: string;
    proofTreeBuildResult: ProofTreeBuildResult;
    augmentedNodesRatio: [number, number];
}

export interface CoqDatasetAugmentedFile {
    filePath: string;
    itemName: string;
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

export type CoqDatasetDirItem = CoqDatasetAugmentedFile | CoqDatasetFolder;

export interface CoqDatasetStats {
    // TODO: Add for how many theorems successfully built proof tree
    augmentedNodesRatio: [number, number];
}

// TODO: Move to utils
export function accumulateStats(
    folderStats: CoqDatasetStats,
    itemStats: CoqDatasetStats
): CoqDatasetStats {
    return {
        augmentedNodesRatio: [
            folderStats.augmentedNodesRatio[0] +
                itemStats.augmentedNodesRatio[0],
            folderStats.augmentedNodesRatio[1] +
                itemStats.augmentedNodesRatio[1],
        ],
    };
}

export function emptyDatasetStats(): CoqDatasetStats {
    return {
        augmentedNodesRatio: [0, 0],
    };
}

export type CoqDataset = CoqDatasetDirItem;
