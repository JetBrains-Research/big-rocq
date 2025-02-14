import * as assert from "assert";
import { readFileSync } from "fs";
import { Range } from "vscode-languageclient";

import { CoqLspClient } from "../coqLsp/coqLspClient";
import {
    CoqParsingError,
    FlecheDocument,
    Goal,
    PpString,
    RangedSpan,
} from "../coqLsp/coqLspTypes";

import { EventLogger } from "../logging/eventLogger";
import { throwOnAbort } from "../utils/abortUtils";
import { getTextInRange } from "../utils/documentUtils";
import { Uri } from "../utils/uri";

import { AllGoals, GoalList, GoalSelector, SingleGoal } from "./goalSelectors";
import { ProofStep, Theorem, TheoremProof, Vernacexpr } from "./parsedTypes";

/**
 * As we have decided that premises = only theorems/definitions
 * with existing proofs, parseCoqFile ignores items without proofs
 * and does not add them into the resulting array.
 */
export async function parseCoqFile(
    uri: Uri,
    client: CoqLspClient,
    abortSignal: AbortSignal,
    extractTheoremInitialGoal: boolean = true,
    eventLogger?: EventLogger
): Promise<Theorem[]> {
    // TODO: Add invariant that no error diags are present in the parsed file
    return client
        .getFlecheDocument(uri)
        .then((doc) => {
            const documentText = readFileSync(uri.fsPath)
                .toString()
                .split("\n");
            return parseFlecheDocument(
                doc,
                documentText,
                client,
                uri,
                abortSignal,
                extractTheoremInitialGoal,
                eventLogger
            );
        })
        .catch((error) => {
            throw new CoqParsingError(
                `failed to parse file with Error: ${error.message}`
            );
        });
}

async function parseFlecheDocument(
    doc: FlecheDocument,
    textLines: string[],
    client: CoqLspClient,
    uri: Uri,
    abortSignal: AbortSignal,
    extractTheoremInitialGoal: boolean,
    eventLogger?: EventLogger
): Promise<Theorem[]> {
    if (doc === null) {
        throw Error("could not parse file");
    }

    const theorems: Theorem[] = [];
    for (let i = 0; i < doc.spans.length; i++) {
        throwOnAbort(abortSignal);

        const span = doc.spans[i];
        try {
            const vernacType = getVernacexpr(getExpr(span));
            if (
                vernacType &&
                [
                    Vernacexpr.VernacDefinition,
                    Vernacexpr.VernacStartTheoremProof,
                ].includes(vernacType)
            ) {
                const thrName = getName(getExpr(span));
                const thrStatement = getTextInRange(
                    doc.spans[i].range.start,
                    doc.spans[i].range.end,
                    textLines,
                    true
                );

                const nextExprVernac = getVernacexpr(getExpr(doc.spans[i + 1]));
                if (i + 1 >= doc.spans.length) {
                    eventLogger?.log(
                        "premise-has-no-proof",
                        `Could not parse the proof in theorem/definition ${thrName}.`
                    );
                } else if (!nextExprVernac) {
                    throw new CoqParsingError("unable to parse proof");
                } else if (
                    ![
                        Vernacexpr.VernacProof,
                        Vernacexpr.VernacAbort,
                        Vernacexpr.VernacEndProof,
                    ].includes(nextExprVernac)
                ) {
                    eventLogger?.log(
                        "premise-has-no-proof",
                        `Could not parse the proof in theorem/definition ${thrName}.`
                    );
                } else {
                    // TODO: Cover with tests, might be a source of bugs if somewhere
                    // absense of initialGoal is not handled properly or invariants are broken
                    let initialGoal: Goal<PpString>[] | null = null;
                    if (extractTheoremInitialGoal) {
                        const goalConfig = await client.getGoalsAtPoint(
                            doc.spans[i + 1].range.start,
                            uri,
                            1
                        );

                        if (goalConfig.err) {
                            throw new CoqParsingError(
                                `unable to get initial goal for theorem: ${thrName}`
                            );
                        }

                        initialGoal = goalConfig.val.goals;
                    }

                    const proof = parseProof(i + 1, doc.spans, textLines);
                    theorems.push(
                        new Theorem(
                            thrName,
                            doc.spans[i].range,
                            thrStatement,
                            proof,
                            initialGoal?.at(0)
                        )
                    );
                }
            }
        } catch (error) {
            // Ignore
        }
    }

    return theorems;
}

function getExpr(span: RangedSpan): any[] | null {
    try {
        return span.span === null ? null : span.span["v"]["expr"][1];
    } catch (error) {
        return null;
    }
}

function getTheoremName(expr: any): string {
    try {
        return expr[2][0][0][0]["v"][1];
    } catch (error) {
        throw new CoqParsingError("invalid theorem name");
    }
}

function getDefinitionName(expr: any): string {
    try {
        return expr[2][0]["v"][1][1];
    } catch (error) {
        throw new CoqParsingError("invalid definition name");
    }
}

function getName(expr: any): string {
    switch (getVernacexpr(expr)) {
        case Vernacexpr.VernacDefinition:
            return getDefinitionName(expr);
        case Vernacexpr.VernacStartTheoremProof:
            return getTheoremName(expr);
        default:
            throw new CoqParsingError(`invalid name for expression: "${expr}"`);
    }
}

function getVernacexpr(expr: any): Vernacexpr | null {
    try {
        return expr[0] as Vernacexpr;
    } catch (error) {
        return null;
    }
}

function getProofEndCommand(expr: { [key: string]: any }): string | null {
    try {
        return expr[1][0];
    } catch (error) {
        return null;
    }
}

function getGoalSelector(expr: any): GoalSelector | undefined {
    try {
        if (hasParallelExecSelector(expr)) {
            return GoalSelector.AllGoals();
        }

        const goalSelectorItem = expr[2][0][2];
        if (goalSelectorItem.length === 0) {
            return undefined;
        }

        const goalSelector = goalSelectorItem[0];
        if (goalSelector === AllGoals.type) {
            return GoalSelector.AllGoals();
        }

        assert(
            goalSelectorItem.length === 1,
            "Expected to have only one goal selector in a tactic, what the hell"
        );

        const selectorType = goalSelector[0];

        switch (selectorType) {
            case SingleGoal.type:
                const goalIndex = goalSelector[1];
                return GoalSelector.SingleGoal(parseInt(goalIndex));
            case GoalList.type:
                // Lists and ranges are represented in the same way
                const rangesOfGoals = mapGoalIndecesToNumbers(goalSelector[1]);
                const listsOfGoals = rangesOfGoals.map((range) => {
                    assert(
                        range.length === 2,
                        "Expected to have a range of two numbers"
                    );
                    return accRange(range[0], range[1]);
                });
                const mergedGoals = mergeGoalSelectorLists(listsOfGoals);
                return GoalSelector.GoalList(mergedGoals);
            default:
                return undefined;
        }
    } catch (error) {
        return undefined;
    }
}

/**
 * "par:" selector is semantically equivalent to "all:" selector
 * but represented differently in the Fleche Document.
 * https://coq.inria.fr/doc/V8.18.0/refman/proof-engine/ltac.html#goal-selectors
 */
function hasParallelExecSelector(expr: any): boolean {
    try {
        if (expr[1]["ext_entry"] === "VernacSolveParallel") {
            return true;
        } else {
            return false;
        }
    } catch (error) {
        return false;
    }
}

/**
 * Input should correctly resolve as an array of arrays of string,
 * where each string is a number. Throws an error if the input is invalid.
 *
 * Example:
 * [
 *  [
 *      "1",
 *      "1"
 *  ],
 *  [
 *      "2",
 *      "3"
 *  ]
 * ]
 */
function mapGoalIndecesToNumbers(goalIndeces: any): number[][] {
    return goalIndeces.map((el: any) => {
        return el.map((el: any) => {
            return parseInt(el);
        });
    });
}

function accRange(from: number, to: number): number[] {
    if (from === to) {
        return [from];
    }

    const result = [];
    for (let i = from; i <= to; i++) {
        result.push(i);
    }
    return result;
}

function mergeGoalSelectorLists(ranges: number[][]): number[] {
    const result = new Set<number>();
    for (const range of ranges) {
        for (const el of range) {
            result.add(el);
        }
    }
    return Array.from(result);
}

function checkIfExprEAdmit(expr: any): boolean {
    try {
        return getProofEndCommand(expr) === "Admitted";
    } catch (error) {
        return false;
    }
}

function parseProof(
    spanIndex: number,
    ast: RangedSpan[],
    lines: string[]
): TheoremProof {
    let index = spanIndex;
    let proven = false;
    const proof: ProofStep[] = [];
    let endPos: Range | null = null;
    let proofContainsAdmit = false;
    let proofHoles: ProofStep[] = [];

    while (!proven && index < ast.length) {
        const span = ast[index];

        const vernacType = getVernacexpr(getExpr(span));
        if (!vernacType) {
            throw new CoqParsingError(
                "unable to derive the vernac type of the sentence"
            );
        }

        if (
            vernacType === Vernacexpr.VernacEndProof ||
            vernacType === Vernacexpr.VernacAbort
        ) {
            const proofStep = new ProofStep(
                getTextInRange(span.range.start, span.range.end, lines),
                vernacType,
                span.range
            );
            proof.push(proofStep);
            proven = true;
            endPos = span.range;

            if (
                checkIfExprEAdmit(getExpr(span)) ||
                vernacType === Vernacexpr.VernacAbort
            ) {
                proofContainsAdmit = true;
            }
        } else {
            const proofText = getTextInRange(
                span.range.start,
                span.range.end,
                lines
            );
            const proofStep = new ProofStep(
                proofText,
                vernacType,
                span.range,
                getGoalSelector(getExpr(span))
            );

            proof.push(proofStep);
            index += 1;

            if (proofText.includes("admit")) {
                proofHoles.push(proofStep);
            }
        }
    }

    if (!proven || endPos === null) {
        throw new CoqParsingError("invalid or incomplete proof");
    }

    const proofObj = new TheoremProof(
        proof,
        endPos,
        proofContainsAdmit,
        proofHoles
    );
    return proofObj;
}
