import { Err, Ok, Result } from "ts-results";
import { Position } from "vscode-languageclient";

import { CoqLspClient } from "../coqLsp/coqLspClient";
import { GoalConfig, PpString } from "../coqLsp/coqLspTypes";

import {
    ProofStep,
    Theorem,
    TheoremProof,
    Vernacexpr,
} from "../coqParser/parsedTypes";
import { unwrapOrThrow } from "../utils/optionWrappers";
import { Uri } from "../utils/uri";

import { CoqProofTree, CoqProofTreeError } from "./coqProofTree";

async function getStateAtPosOrThrow(
    position: Position,
    lspClient: CoqLspClient,
    docUri: Uri,
    docVersion: number
): Promise<GoalConfig<PpString>> {
    const goalAfterTactic = await lspClient.getGoalsAtPoint(
        position,
        docUri,
        docVersion
    );

    if (goalAfterTactic.err) {
        throw new CoqProofTreeError(
            `Unable to get goal at position (${position.line}, ${position.character})`
        );
    }

    return goalAfterTactic.val;
}

async function getStateAfterTactic(
    tactic: ProofStep,
    lspClient: CoqLspClient,
    docUri: Uri,
    docVersion: number
): Promise<GoalConfig<PpString>> {
    return getStateAtPosOrThrow(
        tactic.range.end,
        lspClient,
        docUri,
        docVersion
    );
}

export async function getProofInitialState(
    proof: TheoremProof,
    lspClient: CoqLspClient,
    docUri: Uri,
    docVersion: number
): Promise<GoalConfig<PpString>> {
    return getStateAtPosOrThrow(
        proof.proof_steps[0].range.start,
        lspClient,
        docUri,
        docVersion
    );
}

function throwIfMultipleGoalsOnStart(initialGoals: GoalConfig<PpString>) {
    // TODO: We cannot have multiple goals on start, right?
    if (initialGoals.goals.length !== 1) {
        throw new CoqProofTreeError(
            `Unexpected multiple goals in initial goal in theorem`
        );
    }
}

function throwIfUnexpectedVernacType(step: ProofStep) {
    const expectedVernacTypes = [
        // Focusing
        Vernacexpr.VernacBullet,
        Vernacexpr.VernacSubproof,
        Vernacexpr.VernacEndSubproof,
        // Proof start/end
        Vernacexpr.VernacAbort,
        Vernacexpr.VernacEndProof,
        Vernacexpr.VernacProof,
        Vernacexpr.VernacDefinition,
        Vernacexpr.VernacStartTheoremProof,
        // Tactics
        Vernacexpr.VernacExtend,
        Vernacexpr.VernacPrint,
    ];
    if (!expectedVernacTypes.includes(step.vernac_type)) {
        throw new CoqProofTreeError(
            `Unexpected Vernac type in step: ${step.text}`
        );
    }
}

function throwIfHasGoalSelector(step: ProofStep) {
    // TODO: Figire out how to deal with goal selectors
    // https://coq.inria.fr/doc/V8.18.0/refman/proof-engine/ltac.html#goal-selectors

    const goalSelectorRegex = /^(all:|\d+:|(\d+\s*,\s*)*\d+:|\d+\s*-\s*\d+:)/;
    if (goalSelectorRegex.test(step.text.trim())) {
        throw new CoqProofTreeError(
            `Goal selector found in step: ${step.text}`
        );
    }
}

function throwIfHasZeroTactics(theorem: Theorem) {
    const tacticVernacTypes = [Vernacexpr.VernacExtend];
    const stateChangeTactics = theorem.proof.proof_steps.filter((step) =>
        tacticVernacTypes.includes(step.vernac_type)
    );

    if (stateChangeTactics.length === 0) {
        throw new CoqProofTreeError(`Theorem has no tactics in proof_steps`);
    }
}

export function throwIfCannotProcess(theorem: Theorem) {
    throwIfHasZeroTactics(theorem);
    theorem.proof.proof_steps.forEach((tactic) => {
        throwIfHasGoalSelector(tactic);
        throwIfUnexpectedVernacType(tactic);
    });
}

export async function buildCoqProofTree(
    theorem: Theorem,
    lspClient: CoqLspClient,
    docUri: Uri,
    docVersion: number
): Promise<Result<CoqProofTree, CoqProofTreeError>> {
    try {
        throwIfCannotProcess(theorem);
    } catch (e) {
        if (e instanceof CoqProofTreeError) {
            return Err(e);
        }
    }

    const proof = theorem.proof;
    const initialGoals = await getProofInitialState(
        proof,
        lspClient,
        docUri,
        docVersion
    );
    throwIfMultipleGoalsOnStart(initialGoals);

    const initialGoalNullable = initialGoals.goals.at(0);
    const initialGoal = unwrapOrThrow(initialGoalNullable);

    const proofTree = new CoqProofTree(initialGoal);

    // Refer to algorithm in src/coqProofTree/coqProofTree.ts to
    // understand semantics
    let currentProofState: GoalConfig<PpString> = initialGoals;
    for (const step of proof.proof_steps) {
        if (step.vernac_type === Vernacexpr.VernacEndProof) {
            break;
        }

        try {
            const goalAfterStep = await getStateAfterTactic(
                step,
                lspClient,
                docUri,
                docVersion
            );

            if (step.vernac_type === Vernacexpr.VernacExtend) {
                proofTree.applyToFirstUnsolvedGoal(
                    step,
                    currentProofState,
                    goalAfterStep
                );
            }

            currentProofState = goalAfterStep;
        } catch (e) {
            if (e instanceof CoqProofTreeError) {
                return Err(e);
            }
        }
    }

    return Ok(proofTree);
}
