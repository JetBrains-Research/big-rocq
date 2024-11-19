import { Goal, GoalConfig, PpString } from "../coqLsp/coqLspTypes";

import { ProofStep } from "../coqParser/parsedTypes";
import { unwrapOrThrow } from "../utils/optionWrappers";

export type CoqProofTreeNodeType = Goal<PpString>;
export type CoqProofTreeEdgeType = ProofStep;
export type CoqProofTreeConnectionType = [
    CoqProofTreeNode,
    CoqProofTreeEdgeType,
];

/**
 * Invariant on children array: indices correspond to the order of
 * goals that are produced after applying the tactic
 */
// TODO: Use base class
export class CoqProofTreeNode {
    // Is undefined when it is a root
    parent: CoqProofTreeConnectionType | undefined;

    // Is undefined when there are no more goals
    proofState: CoqProofTreeNodeType | undefined;
    children: CoqProofTreeConnectionType[] = [];

    static createRoot(proofState?: CoqProofTreeNodeType): CoqProofTreeNode {
        return new CoqProofTreeNode(undefined, proofState);
    }

    protected constructor(
        parent?: CoqProofTreeConnectionType,
        proofState?: CoqProofTreeNodeType
    ) {
        this.proofState = proofState;
        if (!parent) {
            this.parent = undefined;
            return;
        }

        this.parent = parent;
        const [parentNode, stepLabel] = parent;
        parentNode.children.push([this, stepLabel]);
    }

    private addChild(
        appliedProofStep: CoqProofTreeEdgeType,
        proofState?: CoqProofTreeNodeType
    ): CoqProofTreeNode {
        return new CoqProofTreeNode([this, appliedProofStep], proofState);
    }

    addChildren(
        appliedProofStep: ProofStep,
        proofState: GoalConfig<PpString>
    ): CoqProofTreeNode[] {
        if (proofState.goals.length === 0) {
            const emptyState = this.addChild(appliedProofStep);
            return [emptyState];
        }

        const newSubgoals: CoqProofTreeNode[] = [];
        for (const goal of proofState.goals) {
            const newChild = this.addChild(appliedProofStep, goal);
            newSubgoals.push(newChild);
        }

        return newSubgoals;
    }

    dfsTraverse(
        callback: (node: CoqProofTreeNode) => void,
        shouldTerminate: (node: CoqProofTreeNode) => boolean = () => false
    ): CoqProofTreeNode | undefined {
        let queue: CoqProofTreeNode[] = [];
        queue.push(this);

        while (queue.length !== 0) {
            const elemNullable = queue.shift();
            const elem = unwrapOrThrow(elemNullable);

            callback(elem);

            if (shouldTerminate(elem)) {
                return elem;
            }

            // We want to traverse children in the same order as
            // they are stored in the children array (actually the
            // order od goals). Therefore the array is reversed before
            // push fronting
            const children = elem.children.map(([child, _]) => child);
            children.reverse().forEach((child) => {
                queue.unshift(child);
            });
        }

        return undefined;
    }

    subtreeFind(
        predicate: (node: CoqProofTreeNode) => boolean
    ): CoqProofTreeNode | undefined {
        return this.dfsTraverse(() => {}, predicate);
    }

    subtreeFilter(
        predicate: (node: CoqProofTreeNode) => boolean
    ): CoqProofTreeNode[] {
        const nodes: CoqProofTreeNode[] = [];
        this.dfsTraverse((node) => {
            if (predicate(node)) {
                nodes.push(node);
            }
        });
        return nodes;
    }

    collectSubtree(): CoqProofTreeNode[] {
        const nodes: CoqProofTreeNode[] = [];
        this.dfsTraverse((node) => nodes.push(node));
        return nodes;
    }

    collectSubtreeEdges(): CoqProofTreeEdgeType[] {
        if (this.isLeaf) {
            return [];
        }

        if (this.children.length === 1) {
            const child = unwrapOrThrow(this.children.at(0));
            const [childNode, edge] = child;
            return [edge].concat(childNode.collectSubtreeEdges());
        } else {
            const firstChild = unwrapOrThrow(this.children.at(0));
            const [_, firstEdge] = firstChild;
            const childrenEdges = this.children.map(([childNode, _]) =>
                childNode.collectSubtreeEdges()
            );
            return [firstEdge].concat(childrenEdges.flat());
        }
    }

    get firstChild(): CoqProofTreeNode | undefined {
        const child = this.children.at(0);
        if (!child) {
            return undefined;
        }

        const [node, _] = child;
        return node;
    }

    get isEmptyState(): boolean {
        return !this.proofState;
    }

    get isLeaf(): boolean {
        return this.children.length === 0;
    }

    get isUnsolvedGoal(): boolean {
        return this.isLeaf && !this.isEmptyState;
    }
}
