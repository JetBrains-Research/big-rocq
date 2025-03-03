import { Severity } from "../logging/eventLogger";

export interface RunParams {
    loggingLevel: Severity;
    coqLspServerPath: string;
    targetRootPath: string;
    workspaceRootPath: string;
    fileAugmentationTimeoutMillis: number;
    fileTypeCheckingTimeoutMillis: number;
    generateDatasetViewer: boolean;
    generateAugmentedCoqFiles: boolean;
    skipZeroProgressTactics: boolean;
}
