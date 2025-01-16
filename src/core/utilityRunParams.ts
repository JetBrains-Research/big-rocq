import { Severity } from "../logging/eventLogger";

export interface RunParams {
    loggingLevel: Severity;
    coqLspServerPath: string;
    targetRootPath: string;
    workspaceRootPath: string;
    theoremValidationTimeoutMillis: number;
    fileTypeCheckingTimeoutMillis: number;
}
