import { expect } from "earl";

import { AllGoals, GoalList, SingleGoal } from "../../coqParser/goalSelectors";
import { ProofStep, Theorem } from "../../coqParser/parsedTypes";
import { parseTheoremsFromCoqFile } from "../commonTestFunctions/coqFileParser";

suite("Coq file parser tests for goal selectors", () => {
    function getTheoremByName(theorems: Theorem[], name: string): Theorem {
        const theorem = theorems.find((th) => th.name === name);
        expect(theorem).not.toBeNullish();
        return theorem!;
    }

    function assertHasNoGoalSelector(proofStep: ProofStep) {
        expect(proofStep.hasGoalSelector).toBeFalsy();
    }

    test("No goal selectors", async () => {
        const doc = await parseTheoremsFromCoqFile(["goal_selectors.v"]);
        const theorem = getTheoremByName(doc, "no_selectors");

        theorem.proof.proof_steps.forEach((step) => {
            assertHasNoGoalSelector(step);
        });
    });

    test("All goal selector", async () => {
        const doc = await parseTheoremsFromCoqFile(["goal_selectors.v"]);
        const theorem = getTheoremByName(doc, "all_selector");

        theorem.proof.proof_steps.forEach((step) => {
            if (step.text === "all: auto.") {
                expect(step.hasGoalSelector).toBeTruthy();
                expect(step.goalSelector instanceof AllGoals).toBeTruthy();
            } else {
                assertHasNoGoalSelector(step);
            }
        });
    });

    test("Single index goal selector", async () => {
        const doc = await parseTheoremsFromCoqFile(["goal_selectors.v"]);
        const theorem = getTheoremByName(doc, "select_nth");

        theorem.proof.proof_steps.forEach((step) => {
            if (step.text === "1: auto.") {
                expect(step.hasGoalSelector).toBeTruthy();
                expect(step.goalSelector instanceof SingleGoal).toBeTruthy();
                if (step.goalSelector instanceof SingleGoal) {
                    expect(step.goalSelector.n).toEqual(1);
                }
            } else {
                assertHasNoGoalSelector(step);
            }
        });
    });

    test("List goal selector", async () => {
        const doc = await parseTheoremsFromCoqFile(["goal_selectors.v"]);
        const theorem = getTheoremByName(doc, "select_list");

        theorem.proof.proof_steps.forEach((step) => {
            if (step.text === "1, 2: simpl; try rewrite IHn; reflexivity.") {
                expect(step.hasGoalSelector).toBeTruthy();
                expect(step.goalSelector instanceof GoalList).toBeTruthy();
                if (step.goalSelector instanceof GoalList) {
                    expect(step.goalSelector.indices).toEqual([1, 2]);
                }
            } else {
                assertHasNoGoalSelector(step);
            }
        });
    });

    test("Complex goal selector", async () => {
        const doc = await parseTheoremsFromCoqFile(["goal_selectors.v"]);
        const theorem = getTheoremByName(doc, "complex_selector");

        theorem.proof.proof_steps.forEach((step) => {
            if (step.text === "1, 2-3: auto.") {
                expect(step.hasGoalSelector).toBeTruthy();
                expect(step.goalSelector instanceof GoalList).toBeTruthy();
                if (step.goalSelector instanceof GoalList) {
                    expect(step.goalSelector.indices).toEqual([1, 2, 3]);
                }
            } else {
                assertHasNoGoalSelector(step);
            }
        });
    });

    test("Parallel goal selector", async () => {
        const doc = await parseTheoremsFromCoqFile(["goal_selectors.v"]);
        const theorem = getTheoremByName(doc, "parallel_selector");

        theorem.proof.proof_steps.forEach((step) => {
            if (step.text === "par: try contradiction; try reflexivity.") {
                expect(step.hasGoalSelector).toBeTruthy();
                expect(step.goalSelector instanceof AllGoals).toBeTruthy();
            } else {
                assertHasNoGoalSelector(step);
            }
        });
    });
});
