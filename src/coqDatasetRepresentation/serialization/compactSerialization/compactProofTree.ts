import * as assert from "assert";
import { Err, Ok, Result } from "ts-results";

// import { CoqProofTree } from "../../../coqProofTree/coqProofTree";
import { CoqProofTreeNode } from "../../../coqProofTree/coqProofTreeNode";
import {
    // constructTheoremWithProof,
    theoremDatasetSampleToString,
} from "../../../coqProofTree/proofBuilder";
import { PpMode, ppGoal } from "../../../utils/proofStatePrinters";
import {
    CoqAugmentedTheoremItem,
    NodeAugmentationResult,
    NodeId,
} from "../../coqDatasetModels";

export interface CompactSerializedCoqProofTreeNode {
    proofState?: string;
    subtreeProof?: Result<string, string>;
    parentEdgeLabel?: string;
    children: CompactSerializedCoqProofTreeNode[];
}

export interface CompactSerializedCoqProofTree {
    root: CompactSerializedCoqProofTreeNode;
}

function compactSerializeCoqProofTreeNode(
    node: CoqProofTreeNode,
    augmentedSamplesMap: Map<NodeId, NodeAugmentationResult>,
    ppType: PpMode,
    theoremName: string,
    parentEdgeLabel: string | undefined
): CompactSerializedCoqProofTreeNode {
    const subtreeProof = augmentedSamplesMap.get(node.index);
    assert(subtreeProof || node.isLeaf);

    const subtreeSampleRes = subtreeProof
        ? subtreeProof.ok
            ? Ok(theoremDatasetSampleToString(subtreeProof.val))
            : Err(subtreeProof.val.message)
        : Err("This is an empty state");

    assert(
        (!node.proofState && node.isLeaf) || (node.proofState && !node.isLeaf)
    );

    return {
        proofState: node.proofState
            ? ppGoal(node.proofState, ppType)
            : undefined,
        subtreeProof: subtreeSampleRes,
        parentEdgeLabel: parentEdgeLabel,
        children: node.children.map(([child, edge]) =>
            compactSerializeCoqProofTreeNode(
                child,
                augmentedSamplesMap,
                ppType,
                theoremName,
                edge.text
            )
        ),
    };
}

export function compactSerializeCoqProofTree(
    proofTreeWithAugmnetation: CoqAugmentedTheoremItem,
    ppType: PpMode,
    theoremName: string
): CompactSerializedCoqProofTree {
    return {
        root: compactSerializeCoqProofTreeNode(
            proofTreeWithAugmnetation.proofTree.root,
            proofTreeWithAugmnetation.samples,
            ppType,
            theoremName,
            undefined
        ),
    };
}
