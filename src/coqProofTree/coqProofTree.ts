import { GoalConfig, PpString } from "../coqLsp/coqLspTypes";

import { ProofStep } from "../coqParser/parsedTypes";

import { CoqProofTreeNode } from "./coqProofTreeNode";

export class CoqProofTreeError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "CoqProofTreeError";
    }
}

// TODO: implement Iterator<CoqProofTreeNode>
export class CoqProofTree {
    constructor(readonly root: CoqProofTreeNode) {}

    // TODO: Rn at Baseline 1 we assume that we cannot focus on
    // smth other than Goal 1.
    applyToFirstUnsolvedGoal(
        appliedProofStep: ProofStep,
        proofState: GoalConfig<PpString>
    ): CoqProofTreeNode[] {
        console.log(`applyToFirstUnsolvedGoal ${appliedProofStep.text}`);
        const firstUnsolvedGoal = this.root.subtreeFind((node) => {
            return node.isUnsolvedGoal;
        });
        if (!firstUnsolvedGoal) {
            throw new CoqProofTreeError(
                "There are no unsolved goals, please explore the proof tree"
            );
        }

        return firstUnsolvedGoal.addChildren(appliedProofStep, proofState);
    }

    bfs(): CoqProofTreeNode[] {
        return this.root.collectSubtree();
    }

    filter(predicate: (value: CoqProofTreeNode) => boolean) {
        const treeItems = this.bfs();
        return treeItems.filter(predicate);
    }

    getStack(): CoqProofTreeNode[] {
        return this.filter((value) => {
            return value.isUnsolvedGoal;
        });
    }
}
