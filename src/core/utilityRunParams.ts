import { Severity } from "../logging/eventLogger";

export interface UtilityRunParams {
    loggingLevel: Severity;
    coqLspServerPath: string;
    targetRootPath: string;
    workspaceRootPath: string;
}

export const defaultUtilityRunParams: UtilityRunParams = {
    loggingLevel: Severity.DEBUG,
    coqLspServerPath:
        "/nix/store/4jw388fmzk419bhkfi6g4cckbcwy1i9h-coq-lsp-0.1.8+8.19/bin/coq-lsp",
    targetRootPath: "dataset/coqpilotTests/src/",
    workspaceRootPath: "dataset/coqpilotTests",
};
