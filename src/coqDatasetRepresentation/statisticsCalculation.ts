import {
    CoqDatasetStats,
    CoqDatasetTheoremItem,
    CoqTheoremStats,
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

    const proofLengthCounts = new Map<number, number>();
    if (proofTreeBuildResult.ok) {
        const samples = proofTreeBuildResult.val.samples;
        for (const sample of samples.values()) {
            if (sample.ok) {
                const proofLength = sample.val.proofLength;
                proofLengthCounts.set(
                    proofLength,
                    (proofLengthCounts.get(proofLength) || 0) + 1
                );
            }
        }
    }

    const stats: CoqTheoremStats = {
        augmentedNodesRatio,
        proofTreeBuildRatio: proofTreeBuildResult.ok ? [1, 1] : [0, 1],
        augmentedProofLengthCounts: proofLengthCounts,
    };

    return {
        parsedTheorem: theorem,
        sourceFilePath: sourceFilePath,
        proofTreeBuildResult: proofTreeBuildResult,
        stats: stats,
    };
}

export function accumulateTheoremStats(
    datasetStats: CoqDatasetStats,
    theoremStats: CoqTheoremStats
): CoqDatasetStats {
    return {
        augmentedNodesRatio: ratioAddition(
            theoremStats.augmentedNodesRatio,
            datasetStats.augmentedNodesRatio
        ),
        proofTreeBuildRatio: ratioAddition(
            theoremStats.proofTreeBuildRatio,
            datasetStats.proofTreeBuildRatio
        ),
        augmentedProofLengthCounts: accumulateProofLengthCounts(
            theoremStats.augmentedProofLengthCounts,
            datasetStats.augmentedProofLengthCounts
        ),
        locChangeAfterAugmentation: datasetStats.locChangeAfterAugmentation,
        processingTimeMillis: datasetStats.processingTimeMillis,
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
        augmentedProofLengthCounts: accumulateProofLengthCounts(
            stats1.augmentedProofLengthCounts,
            stats2.augmentedProofLengthCounts
        ),
        locChangeAfterAugmentation: ratioAddition(
            stats1.locChangeAfterAugmentation,
            stats2.locChangeAfterAugmentation
        ),
        processingTimeMillis:
            stats1.processingTimeMillis + stats2.processingTimeMillis,
    };
}

export function computeAverageProofLength(stats: CoqDatasetStats): number {
    if (stats.augmentedProofLengthCounts.size === 0) {
        return 0;
    }

    let totalLength = 0;
    let totalCount = 0;
    for (const [length, count] of stats.augmentedProofLengthCounts) {
        totalLength += length * count;
        totalCount += count;
    }

    const average = totalLength / totalCount;
    const rounded = Math.round(average * 100) / 100;

    return rounded;
}

export function emptyDatasetStats(): CoqDatasetStats {
    return {
        augmentedNodesRatio: emptyRatio,
        proofTreeBuildRatio: emptyRatio,
        augmentedProofLengthCounts: new Map(),
        locChangeAfterAugmentation: [0, 0],
        processingTimeMillis: 0,
    };
}

const emptyRatio: Ratio = [0, 0];

function accumulateProofLengthCounts(
    counts1: Map<number, number>,
    counts2: Map<number, number>
): Map<number, number> {
    const result = new Map(counts1);
    for (const [length, count] of counts2) {
        result.set(length, (result.get(length) || 0) + count);
    }
    return result;
}

function ratioAddition(ratio1: Ratio, ratio2: Ratio): Ratio {
    return [ratio1[0] + ratio2[0], ratio1[1] + ratio2[1]];
}
