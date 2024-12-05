import { Goal, Hyp, PpString } from "../coqLsp/coqLspTypes";

import { ProofStep, Theorem } from "../coqParser/parsedTypes";

export interface TheoremDatasetSample {
    theoremStatement: string;
    proof: string;
}

export function theoremDatasetSampleToString(
    sample: TheoremDatasetSample,
): string {
    return `${sample.theoremStatement}\nProof.\n${sample.proof}\nQed.`;
}

function hypToString(hyp: Hyp<PpString>): string {
    return `${hyp.names.join(" ")} : ${hyp.ty}`;
}

function goalToTheoremStatement(
    proofGoal: Goal<PpString>, 
    currentTheoremName: string, 
    existingTheorems: string[],
): string {
    const auxTheoremConcl = proofGoal?.ty;
    const theoremIndeces = proofGoal?.hyps
        .map((hyp) => `(${hypToString(hyp)})`)
        .join(" ");

    let name = existingTheorems.includes(currentTheoremName) ? `${currentTheoremName}_helper` : currentTheoremName;
    const random = Math.floor(Math.random() * 1000);
    name = existingTheorems.includes(name) ? `${name}_${random}` : name;

    return `Lemma ${name} ${theoremIndeces} :\n   ${auxTheoremConcl}.`;
}

export function constructTheoremWithProof(
    proofState: Goal<PpString>,
    proofSteps: ProofStep[],
    currentTheoremName: string,
    existingTheorems: Theorem[]
): TheoremDatasetSample {
    const theoremNames = existingTheorems.map((theorem) => theorem.name);
    const theoremStatement = goalToTheoremStatement(proofState, currentTheoremName, theoremNames);
    const proof = proofSteps.map((step) => step.text).join("\n");
    return { theoremStatement, proof };
}
