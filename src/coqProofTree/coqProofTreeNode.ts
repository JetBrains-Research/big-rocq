import { Goal, GoalConfig, PpString } from "../coqLsp/coqLspTypes";

import { ProofStep } from "../coqParser/parsedTypes";
import { unwrapOrThrow } from "../utils/optionWrappers";
import { NodeIndexer } from "./nodeIndexer";
import * as assert from "assert";

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

    static createRoot(
        indexer: NodeIndexer,
        proofState?: CoqProofTreeNodeType,
    ): CoqProofTreeNode {
        return new CoqProofTreeNode(indexer.next, undefined, proofState);
    }

    protected constructor(
        readonly index: number,
        parent?: CoqProofTreeConnectionType,
        proofState?: CoqProofTreeNodeType,
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
        indexer: NodeIndexer,
        appliedProofStep: CoqProofTreeEdgeType,
        proofState?: CoqProofTreeNodeType
    ): CoqProofTreeNode {
        return new CoqProofTreeNode(indexer.next, [this, appliedProofStep], proofState);
    }

    // TODO: Prove correctness of approach for Baseline 1
    // Destruct number of goals before and after tactic application: (B, A)
    // case (B = A): create edge B_0 -> A_0; break
    // case (B - 1 = A): create edge B_0 -> []; break
    // case (B + k = A): /* Means tactic applied is kind of destruct */ create edges for i in 0..k B_0 -> A_k
    addChildren(
        indexer: NodeIndexer,
        appliedProofStep: ProofStep,
        proofStateBeforeTactic: GoalConfig<PpString>,
        proofStateAfterTactic: GoalConfig<PpString>,
    ): CoqProofTreeNode[] {
        if (proofStateBeforeTactic.goals.length - 1 === proofStateAfterTactic.goals.length) {
            // case A == B - 1
            const emptyState = this.addChild(indexer, appliedProofStep);

            return [emptyState];
        } else if (proofStateBeforeTactic.goals.length <= proofStateAfterTactic.goals.length) {
            // case B + k = A (k possibly 0)
            const k = proofStateAfterTactic.goals.length - proofStateBeforeTactic.goals.length + 1;

            const newSubgoals: CoqProofTreeNode[] = [];
            for (let i = 0; i < k; i++) {
                const goal = proofStateAfterTactic.goals[i];
                const newChild = this.addChild(indexer, appliedProofStep, goal);
                newSubgoals.push(newChild);
            }

            return newSubgoals;
        } 

        assert(false, `Invariant for adding new node in baseline 1 broken for tactic ${appliedProofStep.text}`);
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

    get isRoot(): boolean {
        return this.parent === undefined;
    }
}
