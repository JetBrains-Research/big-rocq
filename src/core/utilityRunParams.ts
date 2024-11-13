import { Severity } from "../logging/eventLogger";

export interface UtilityRunParams {
    loggingLevel: Severity;
    coqLspServerPath: string;
}

export const defaultUtilityRunParams: UtilityRunParams = {
    loggingLevel: Severity.DEBUG,
    coqLspServerPath: "coq-lsp",
};
