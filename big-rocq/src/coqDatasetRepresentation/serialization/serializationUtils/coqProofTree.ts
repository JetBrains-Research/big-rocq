import { CoqProofTree } from "../../../coqProofTree/coqProofTree";
import { CoqProofTreeNode } from "../../../coqProofTree/coqProofTreeNode";

import { SerializedGoal, serializeGoal } from "./goalParser";
import { SerializedProofStep, serializeProofStep } from "./theoremData";

export interface SerializedCoqProofTreeNode {
    proofState?: SerializedGoal;
    children: SerializedCoqProofTreeConnection[];
}

export interface SerializedCoqProofTreeConnection {
    child: SerializedCoqProofTreeNode;
    edge: SerializedProofStep;
}

export interface SerializedCoqProofTree {
    root: SerializedCoqProofTreeNode;
}

function serializeCoqProofTreeNode(
    node: CoqProofTreeNode
): SerializedCoqProofTreeNode {
    return {
        proofState: node.proofState
            ? serializeGoal(node.proofState)
            : undefined,
        children: node.children.map(([child, edge]) => ({
            child: serializeCoqProofTreeNode(child),
            edge: serializeProofStep(edge),
        })),
    };
}

export function serializeCoqProofTree(
    tree: CoqProofTree
): SerializedCoqProofTree {
    return {
        root: serializeCoqProofTreeNode(tree.root),
    };
}
