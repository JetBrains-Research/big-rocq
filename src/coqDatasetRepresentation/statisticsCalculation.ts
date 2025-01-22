import {
    CoqDatasetStats,
    CoqDatasetTheoremItem,
    ProofTreeBuildResult,
    Ratio,
} from "../coqDatasetRepresentation/coqDatasetModels";
import { Theorem } from "../coqParser/parsedTypes";

import { calculateSuccessfullyAugmentedNodes } from "./coqDatasetUtils";

export function createTheoremWithStats(
    theorem: Theorem,
    sourceFilePath: string,
    proofTreeBuildResult: ProofTreeBuildResult
): CoqDatasetTheoremItem {
    const augmentedNodesRatio = proofTreeBuildResult.ok
        ? calculateSuccessfullyAugmentedNodes(proofTreeBuildResult.val)
        : emptyRatio;

    const stats: CoqDatasetStats = {
        augmentedNodesRatio,
        proofTreeBuildRatio: proofTreeBuildResult.ok ? [1, 1] : [0, 1],
    };

    return {
        parsedTheorem: theorem,
        sourceFilePath: sourceFilePath,
        proofTreeBuildResult: proofTreeBuildResult,
        stats: stats,
    };
}

export function accumulateStats(
    stats1: CoqDatasetStats,
    stats2: CoqDatasetStats
): CoqDatasetStats {
    return {
        augmentedNodesRatio: ratioAddition(
            stats1.augmentedNodesRatio,
            stats2.augmentedNodesRatio
        ),
        proofTreeBuildRatio: ratioAddition(
            stats1.proofTreeBuildRatio,
            stats2.proofTreeBuildRatio
        ),
    };
}

function ratioAddition(ratio1: Ratio, ratio2: Ratio): Ratio {
    return [ratio1[0] + ratio2[0], ratio1[1] + ratio2[1]];
}

export function emptyDatasetStats(): CoqDatasetStats {
    return {
        augmentedNodesRatio: emptyRatio,
        proofTreeBuildRatio: emptyRatio,
    };
}

const emptyRatio: Ratio = [0, 0];
