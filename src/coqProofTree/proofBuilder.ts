import * as assert from "assert";

import { Goal, Hyp, PpString } from "../coqLsp/coqLspTypes";

import { TheoremDatasetSample } from "../coqDatasetRepresentation/coqDatasetModels";
import { ProofStep } from "../coqParser/parsedTypes";

import { CoqProofTree } from "./coqProofTree";
import { HypothesesDict } from "./coqTheoremValidator";

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
    predefinedIndex: number,
    skippedHyps: HypothesesDict
): string {
    const auxTheoremConcl = proofGoal?.ty;
    const theoremIndeces = removeSkippedHypotheses(proofGoal.hyps, skippedHyps)
        .map((hyp) => `(${hypToString(hyp)})`)
        .join(" ");

    let name = `${currentTheoremName}_br_helper_${predefinedIndex}`;
    return `Theorem ${name} ${theoremIndeces} :\n   ${auxTheoremConcl}.`;
}

function removeSkippedHypotheses(
    theoremHyps: Hyp<PpString>[],
    skippedHyps: HypothesesDict
): Hyp<PpString>[] {
    const newHyps: Hyp<PpString>[] = [];
    for (const hyp of theoremHyps) {
        const filteredNames = hyp.names.filter(
            (name) => skippedHyps.get(name as string) === undefined
        );
        if (filteredNames.length > 0) {
            newHyps.push({ names: filteredNames, ty: hyp.ty });
        }
    }

    return newHyps;
}

export function constructTheoremWithProof(
    proofState: Goal<PpString>,
    proofSteps: ProofStep[],
    currentTheoremName: string,
    predefinedIndex: number,
    skippedHyps: HypothesesDict
): TheoremDatasetSample {
    const theoremStatement = goalToTheoremStatement(
        proofState,
        currentTheoremName,
        predefinedIndex,
        skippedHyps
    );
    const proof = proofSteps.map((step) => step.text).join("\n");
    return { theoremStatement, proof, proofLength: proofSteps.length };
}

export function augmentTreeToSamples(
    coqProofTree: CoqProofTree,
    currentTheoremName: string,
    skippedHyps: HypothesesDict
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
                node.index,
                skippedHyps
            );
            samples.set(node.index, sample);
        } else {
            assert(node.isLeaf);
        }
    }

    return samples;
}
