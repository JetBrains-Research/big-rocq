import { CoqProofTree } from "../../../coqProofTree/coqProofTree";
import { CoqProofTreeNode } from "../../../coqProofTree/coqProofTreeNode";
import {
    constructTheoremWithProof,
    theoremDatasetSampleToString,
} from "../../../coqProofTree/proofBuilder";
import { PpMode, ppGoal } from "../../../utils/proofStatePrinters";

export interface CompactSerializedCoqProofTreeNode {
    proofState?: string;
    subtreeProof?: string;
    parentEdgeLabel?: string;
    children: CompactSerializedCoqProofTreeNode[];
}

export interface CompactSerializedCoqProofTree {
    root: CompactSerializedCoqProofTreeNode;
}

function getSubtreeProof(
    treeNode: CoqProofTreeNode,
    theoremName: string
): string | undefined {
    if (treeNode.proofState) {
        const sample = constructTheoremWithProof(
            treeNode.proofState,
            treeNode.collectSubtreeEdges(),
            theoremName
        );
        return theoremDatasetSampleToString(sample);
    } else {
        return undefined;
    }
}

function compactSerializeCoqProofTreeNode(
    node: CoqProofTreeNode,
    ppType: PpMode,
    theoremName: string,
    parentEdgeLabel: string | undefined
): CompactSerializedCoqProofTreeNode {
    return {
        proofState: node.proofState
            ? ppGoal(node.proofState, ppType)
            : undefined,
        subtreeProof: getSubtreeProof(node, theoremName),
        parentEdgeLabel: parentEdgeLabel,
        children: node.children.map(([child, edge]) =>
            compactSerializeCoqProofTreeNode(
                child,
                ppType,
                theoremName,
                edge.text
            )
        ),
    };
}

export function compactSerializeCoqProofTree(
    tree: CoqProofTree,
    ppType: PpMode,
    theoremName: string
): CompactSerializedCoqProofTree {
    return {
        root: compactSerializeCoqProofTreeNode(
            tree.root,
            ppType,
            theoremName,
            undefined
        ),
    };
}
