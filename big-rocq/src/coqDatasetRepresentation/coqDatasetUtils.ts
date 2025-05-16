import * as assert from "assert";

import { CoqAugmentedTheoremItem } from "./coqDatasetModels";

export function calculateSuccessfullyAugmentedNodes(
    proofTreeWithAugmnetation: CoqAugmentedTheoremItem
): [number, number] {
    const samples = proofTreeWithAugmnetation.samples;
    const nodes = proofTreeWithAugmnetation.proofTree.dfs();

    let total = 0;
    let successful = 0;
    for (const node of nodes) {
        // If leaf, would always be not augmented
        if (!node.isLeaf) {
            assert(samples.has(node.index));

            // If root, would always be successfull
            if (!node.isRoot) {
                total++;
                if (samples.get(node.index)?.ok) {
                    successful++;
                }
            }
        }
    }

    return [successful, total];
}
