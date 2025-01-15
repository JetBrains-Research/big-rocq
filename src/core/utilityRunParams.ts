import { Severity } from "../logging/eventLogger";

export interface UtilityRunParams {
    loggingLevel: Severity;
    coqLspServerPath: string;
    targetRootPath: string;
    workspaceRootPath: string;
}

export const defaultUtilityRunParams: UtilityRunParams = {
    loggingLevel: Severity.WARNING,
    coqLspServerPath:
        "/Users/andrei/.opam/big-rocq/bin/coq-lsp",
    targetRootPath: "dataset/hahn/",
    // TODO: Handle not absolute path or leave comment that it is required 
    workspaceRootPath: "/Users/andrei/MCS_Projects/coqPilotOther/bigRocq/dataset/hahn",
};
