import * as assert from "assert";

import { Goal, Hyp, PpString } from "../coqLsp/coqLspTypes";

import { TheoremDatasetSample } from "../coqDatasetRepresentation/coqDatasetModels";
import { ProofStep } from "../coqParser/parsedTypes";

import { CoqProofTree } from "./coqProofTree";

export function theoremDatasetSampleToString(
    sample: TheoremDatasetSample
): string {
    return `${sample.theoremStatement}\nProof.\n${sample.proof}\nQed.`;
}

function hypToString(hyp: Hyp<PpString>): string {
    return `${hyp.names.join(" ")} : ${hyp.ty}`;
}

function goalToTheoremStatement(
    proofGoal: Goal<PpString>,
    currentTheoremName: string,
    predefinedIndex: number
): string {
    const auxTheoremConcl = proofGoal?.ty;
    const theoremIndeces = proofGoal?.hyps
        .map((hyp) => `(${hypToString(hyp)})`)
        .join(" ");

    let name = `${currentTheoremName}_helper_${predefinedIndex}`;
    return `Theorem ${name} ${theoremIndeces} :\n   ${auxTheoremConcl}.`;
}

export function constructTheoremWithProof(
    proofState: Goal<PpString>,
    proofSteps: ProofStep[],
    currentTheoremName: string,
    predefinedIndex: number
): TheoremDatasetSample {
    const theoremStatement = goalToTheoremStatement(
        proofState,
        currentTheoremName,
        predefinedIndex
    );
    const proof = proofSteps.map((step) => step.text).join("\n");
    return { theoremStatement, proof, proofLength: proofSteps.length };
}

export function augmentTreeToSamples(
    coqProofTree: CoqProofTree,
    currentTheoremName: string
): Map<number, TheoremDatasetSample> {
    const nodes = coqProofTree.dfs();

    const samples: Map<number, TheoremDatasetSample> = new Map();
    for (const node of nodes) {
        if (node.proofState) {
            assert(!node.isLeaf);
            const sample = constructTheoremWithProof(
                node.proofState,
                node.collectSubtreeEdges(),
                currentTheoremName,
                node.index
            );
            samples.set(node.index, sample);
        } else {
            assert(node.isLeaf);
        }
    }

    return samples;
}
