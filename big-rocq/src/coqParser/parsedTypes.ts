/* eslint-disable @typescript-eslint/naming-convention */
import { Range } from "vscode-languageclient";

import { Goal, PpString } from "../coqLsp/coqLspTypes";

import { GoalSelector } from "./goalSelectors";

export enum Vernacexpr {
    VernacLoad = "VernacLoad",
    VernacSyntaxExtension = "VernacSyntaxExtension",
    VernacOpenCloseScope = "VernacOpenCloseScope",
    VernacDelimiters = "VernacDelimiters",
    VernacBindScope = "VernacBindScope",
    VernacInfix = "VernacInfix",
    VernacNotation = "VernacNotation",
    VernacNotationAddFormat = "VernacNotationAddFormat",
    VernacDefinition = "VernacDefinition",
    VernacStartTheoremProof = "VernacStartTheoremProof",
    VernacEndProof = "VernacEndProof",
    VernacExactProof = "VernacExactProof",
    VernacAssumption = "VernacAssumption",
    VernacInductive = "VernacInductive",
    VernacFixpoint = "VernacFixpoint",
    VernacCoFixpoint = "VernacCoFixpoint",
    VernacScheme = "VernacScheme",
    VernacCombinedScheme = "VernacCombinedScheme",
    VernacUniverse = "VernacUniverse",
    VernacConstraint = "VernacConstraint",
    VernacBeginSection = "VernacBeginSection",
    VernacEndSegment = "VernacEndSegment",
    VernacRequire = "VernacRequire",
    VernacImport = "VernacImport",
    VernacCanonical = "VernacCanonical",
    VernacCoercion = "VernacCoercion",
    VernacIdentityCoercion = "VernacIdentityCoercion",
    VernacNameSectionHypSet = "VernacNameSectionHypSet",
    VernacInstance = "VernacInstance",
    VernacContext = "VernacContext",
    VernacDeclareInstances = "VernacDeclareInstances",
    VernacDeclareClass = "VernacDeclareClass",
    VernacDeclareModule = "VernacDeclareModule",
    VernacDefineModule = "VernacDefineModule",
    VernacDeclareModuleType = "VernacDeclareModuleType",
    VernacInclude = "VernacInclude",
    VernacSolveExistential = "VernacSolveExistential",
    VernacAddLoadPath = "VernacAddLoadPath",
    VernacRemoveLoadPath = "VernacRemoveLoadPath",
    VernacAddMLPath = "VernacAddMLPath",
    VernacDeclareMLModule = "VernacDeclareMLModule",
    VernacChdir = "VernacChdir",
    VernacWriteState = "VernacWriteState",
    VernacRestoreState = "VernacRestoreState",
    VernacResetName = "VernacResetName",
    VernacResetInitial = "VernacResetInitial",
    VernacBack = "VernacBack",
    VernacBackTo = "VernacBackTo",
    VernacCreateHintDb = "VernacCreateHintDb",
    VernacRemoveHints = "VernacRemoveHints",
    VernacHints = "VernacHints",
    VernacSyntacticDefinition = "VernacSyntacticDefinition",
    VernacDeclareImplicits = "VernacDeclareImplicits",
    VernacArguments = "VernacArguments",
    VernacArgumentsScope = "VernacArgumentsScope",
    VernacReserve = "VernacReserve",
    VernacGeneralizable = "VernacGeneralizable",
    VernacSetOpacity = "VernacSetOpacity",
    VernacSetStrategy = "VernacSetStrategy",
    VernacUnsetOption = "VernacUnsetOption",
    VernacSetOption = "VernacSetOption",
    VernacAddOption = "VernacAddOption",
    VernacRemoveOption = "VernacRemoveOption",
    VernacMemOption = "VernacMemOption",
    VernacPrintOption = "VernacPrintOption",
    VernacCheckMayEval = "VernacCheckMayEval",
    VernacGlobalCheck = "VernacGlobalCheck",
    VernacDeclareReduction = "VernacDeclareReduction",
    VernacPrint = "VernacPrint",
    VernacSearch = "VernacSearch",
    VernacLocate = "VernacLocate",
    VernacRegister = "VernacRegister",
    VernacComments = "VernacComments",
    VernacAbort = "VernacAbort",
    VernacAbortAll = "VernacAbortAll",
    VernacRestart = "VernacRestart",
    VernacUndo = "VernacUndo",
    VernacUndoTo = "VernacUndoTo",
    VernacBacktrack = "VernacBacktrack",
    VernacFocus = "VernacFocus",
    VernacUnfocus = "VernacUnfocus",
    VernacUnfocused = "VernacUnfocused",
    VernacBullet = "VernacBullet",
    VernacSubproof = "VernacSubproof",
    VernacEndSubproof = "VernacEndSubproof",
    VernacShow = "VernacShow",
    VernacCheckGuard = "VernacCheckGuard",
    VernacProof = "VernacProof",
    VernacProofMode = "VernacProofMode",
    VernacToplevelControl = "VernacToplevelControl",
    VernacExtend = "VernacExtend",
}

export class ProofStep {
    constructor(
        public text: string,
        public vernac_type: Vernacexpr,
        public range: Range,
        public goalSelector?: GoalSelector
    ) {}

    public toString(): string {
        return this.text;
    }

    get hasGoalSelector(): boolean {
        return this.goalSelector !== undefined;
    }
}

export class TheoremProof {
    constructor(
        public proof_steps: ProofStep[],
        public end_pos: Range,
        public is_incomplete: boolean,
        public holes: ProofStep[]
    ) {}

    public toString(): string {
        let text = "";
        for (const step of this.proof_steps) {
            text +=
                step.toString() +
                (step.vernac_type !== Vernacexpr.VernacBullet ? "\n" : " ");
        }
        return text;
    }

    /**
     * @param contentsOnly Whether to return only the inner contents of the proof,
     * without 'Proof.' and 'Qed.'
     */
    public onlyText(contentsOnly: boolean = false): string {
        let text = "";
        const proofSteps = contentsOnly
            ? this.proof_steps.slice(1, -1)
            : this.proof_steps;
        for (const step of proofSteps) {
            text +=
                step.text.trim() +
                (step.vernac_type !== Vernacexpr.VernacBullet ? "\n" : " ");
        }
        return text;
    }
}

export class Theorem {
    constructor(
        public name: string,
        public statement_range: Range,
        public statement: string,
        public proof: TheoremProof,
        public initial_goal: Goal<PpString> | null = null
    ) {}

    public toString(): string {
        let text = this.statement;
        text += "\n" + this.proof.toString();
        return text;
    }

    public onlyText(): string {
        let text = this.statement;
        text += "\n" + this.proof.onlyText();
        return text;
    }
}
