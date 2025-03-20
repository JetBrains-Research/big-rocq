import Ajv, { DefinedError, JSONSchemaType } from "ajv";

import { RunParams } from "../core/utilityRunParams";
import { Severity } from "../logging/eventLogger";
import { stringifyAnyValue, stringifyDefinedValue } from "../utils/printers";

import { BigRocqCliArguments } from "./args";

export class ArgValidationError extends Error {
    constructor(errorMessage: string) {
        super(errorMessage);
    }
}

export const bigRocqRunParamsSchema: JSONSchemaType<BigRocqCliArguments> = {
    type: "object",
    properties: {
        targetRootPath: { type: "string", nullable: false },
        workspaceRootPath: { type: "string", nullable: false },
        coqLspServerPath: { type: "string", nullable: false },
        fileAugmentationTimeout: { type: "number", nullable: false },
        fileTimeout: { type: "number", nullable: false },
        generateDatasetViewer: { type: "boolean", nullable: false },
        generateAugmentedCoqFiles: { type: "boolean", nullable: false },
        skipZeroProgressTactics: { type: "boolean", nullable: false },
        version: { type: "boolean", nullable: true },
        verbose: { type: "boolean", nullable: true },
        debug: { type: "boolean", nullable: true },
        help: { type: "boolean", nullable: true },
    },
    required: [
        "targetRootPath",
        "workspaceRootPath",
        "coqLspServerPath",
        "fileAugmentationTimeout",
        "fileTimeout",
        "generateDatasetViewer",
        "generateAugmentedCoqFiles",
        "skipZeroProgressTactics",
    ],
    additionalProperties: false,
};

export function parseUtilityRunParams(runArgs: BigRocqCliArguments): RunParams {
    const ajv = new Ajv();
    const validatedArgs = validateAndParseJson(
        runArgs,
        bigRocqRunParamsSchema,
        ajv
    );

    const logLevel = validatedArgs.debug
        ? Severity.DEBUG
        : validatedArgs.verbose
          ? Severity.INFO
          : Severity.WARNING;
    return {
        loggingLevel: logLevel,
        coqLspServerPath: validatedArgs.coqLspServerPath,
        targetRootPath: validatedArgs.targetRootPath,
        workspaceRootPath: validatedArgs.workspaceRootPath,
        fileAugmentationTimeoutMillis: validatedArgs.fileAugmentationTimeout,
        fileTypeCheckingTimeoutMillis: validatedArgs.fileTimeout,
        generateDatasetViewer: validatedArgs.generateDatasetViewer,
        generateAugmentedCoqFiles: validatedArgs.generateAugmentedCoqFiles,
        skipZeroProgressTactics: validatedArgs.skipZeroProgressTactics,
    };
}

function validateAndParseJson<T>(
    json: any,
    targetClassSchema: JSONSchemaType<T>,
    jsonSchemaValidator: Ajv
): T {
    const instance: T = json as T;
    const validate = jsonSchemaValidator.compile(targetClassSchema);
    if (!validate(instance)) {
        const settingsName = targetClassSchema.title;
        if (settingsName === undefined) {
            throw Error(
                `specified \`targetClassSchema\` does not have \`title\`; while resolving json: ${stringifyAnyValue(json)}`
            );
        }
        const ajvErrors = validate.errors as DefinedError[];
        if (ajvErrors === null || ajvErrors === undefined) {
            throw Error(
                `validation with Ajv failed, but \`validate.errors\` are not defined; while resolving json: ${stringifyAnyValue(json)}`
            );
        }
        throw new ArgValidationError(
            `unable to validate json ${stringifyAnyValue(json)}: ${stringifyDefinedValue(validate.errors)}`
        );
    }
    return instance;
}
