import { GoalConfig, PpString } from "../coqLsp/coqLspTypes";

import { ProofStep } from "../coqParser/parsedTypes";

import { CoqProofTreeNode, CoqProofTreeNodeType } from "./coqProofTreeNode";
import { NodeIndexer } from "./nodeIndexer";

export class CoqProofTreeError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "CoqProofTreeError";
    }
}

// TODO: implement Iterator<CoqProofTreeNode>
export class CoqProofTree {
    root: CoqProofTreeNode;
    nodeIndexer: NodeIndexer = new NodeIndexer();

    constructor(proofState?: CoqProofTreeNodeType) {
        this.root = CoqProofTreeNode.createRoot(this.nodeIndexer, proofState);
    }

    // TODO: Rn at Baseline 1 we assume that we cannot focus on
    // smth other than Goal 1.

    // TODO: Say we have m goals before tactic application
    // After tactic application in Baseline 1 I assume that 
    // we will never have less than m - 1 goals
    // Well, under assumtion that tactic (in baseline 1) is applied 
    // to no more than 1 goal, that should be correct. 

    applyToFirstUnsolvedGoal(
        appliedProofStep: ProofStep,
        proofStateBeforeTactic: GoalConfig<PpString>,
        proofStateAfterTactic: GoalConfig<PpString>,
    ): CoqProofTreeNode[] {
        const firstUnsolvedGoal = this.root.subtreeFind((node) => {
            return node.isUnsolvedGoal;
        });
        if (!firstUnsolvedGoal) {
            throw new CoqProofTreeError(
                "There are no unsolved goals, please explore the proof tree"
            );
        }

        return firstUnsolvedGoal.addChildren(this.nodeIndexer, appliedProofStep, proofStateBeforeTactic, proofStateAfterTactic);
    }

    dfs(): CoqProofTreeNode[] {
        return this.root.collectSubtree();
    }

    filter(predicate: (value: CoqProofTreeNode) => boolean) {
        const treeItems = this.dfs();
        return treeItems.filter(predicate);
    }

    getStack(): CoqProofTreeNode[] {
        return this.filter((value) => {
            return value.isUnsolvedGoal;
        });
    }
}
