import { Goal, Hyp, PpString } from "../coqLsp/coqLspTypes";

import { ProofStep } from "../coqParser/parsedTypes";

export interface TheoremDatasetSample {
    theoremStatement: string;
    proof: string;
}

export function theoremDatasetSampleToString(
    sample: TheoremDatasetSample
): string {
    return `${sample.theoremStatement}\nProof.\n${sample.proof}\nQed.`;
}

function hypToString(hyp: Hyp<PpString>): string {
    return `${hyp.names.join(" ")} : ${hyp.ty}`;
}

function goalToTheoremStatement(proofGoal: Goal<PpString>): string {
    const auxTheoremConcl = proofGoal?.ty;
    const theoremIndeces = proofGoal?.hyps
        .map((hyp) => `(${hypToString(hyp)})`)
        .join(" ");
    return `Lemma helper_theorem ${theoremIndeces} :\n   ${auxTheoremConcl}.`;
}

export function constructTheoremWithProof(
    proofState: Goal<PpString>,
    proofSteps: ProofStep[]
): TheoremDatasetSample {
    const theoremStatement = goalToTheoremStatement(proofState);
    const proof = proofSteps.map((step) => step.text).join("\n");
    return { theoremStatement, proof };
}
