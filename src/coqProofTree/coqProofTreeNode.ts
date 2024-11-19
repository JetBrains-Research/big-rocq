import { Goal, GoalConfig, PpString } from "../coqLsp/coqLspTypes";

import { ProofStep } from "../coqParser/parsedTypes";
import { unwrapOrThrow } from "../utils/optionWrappers";

export type CoqProofTreeNodeType = Goal<PpString>;
export type CoqProofTreeEdgeType = ProofStep;

/**
 * Invariant on children array: indices correspond to the order of
 * goals that are produced after applying the tactic
 */
// TODO: Use base class
export class CoqProofTreeNode {
    // Is undefined when it is a root
    parent: CoqProofTreeNode | undefined;

    // Is undefined when there are no more goals
    proofState: CoqProofTreeNodeType | undefined;
    children: [CoqProofTreeNode, CoqProofTreeEdgeType][] = [];

    static createRoot(proofState?: CoqProofTreeNodeType): CoqProofTreeNode {
        return new CoqProofTreeNode(undefined, proofState);
    }

    protected constructor(
        parent?: [CoqProofTreeNode, CoqProofTreeEdgeType],
        proofState?: CoqProofTreeNodeType
    ) {
        this.proofState = proofState;
        if (!parent) {
            this.parent = undefined;
            return;
        }

        const [parentNode, stepLabel] = parent;
        this.parent = parentNode;
        this.parent.children.push([this, stepLabel]);
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

    // TODO: Refactor dublicating dfs
    subtreeFind(
        predicate: (node: CoqProofTreeNode) => boolean
    ): CoqProofTreeNode | undefined {
        let queue: CoqProofTreeNode[] = [];
        queue.push(this);

        while (queue.length !== 0) {
            const elemNullable = queue.shift();
            const elem = unwrapOrThrow(elemNullable);
            if (predicate(elem)) {
                return elem;
            }

            elem.children.forEach((child, _) => {
                const [childNode, _edge] = child;
                queue.unshift(childNode);
            });
        }

        return undefined;
    }

    collectSubtree(): CoqProofTreeNode[] {
        let nodes: CoqProofTreeNode[] = [];
        let queue: CoqProofTreeNode[] = [];
        queue.push(this);

        while (queue.length !== 0) {
            const elemNullable = queue.shift();
            const elem = unwrapOrThrow(elemNullable);
            nodes.push(elem);

            elem.children.forEach((child, _) => {
                const [childNode, _edge] = child;
                queue.unshift(childNode);
            });
        }

        return nodes;
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
