import { parse } from "ts-command-line-args";

export interface BigRocqCliArguments {
    targetRootPath: string;
    workspaceRootPath: string;
    coqLspServerPath: string;
    theoremAugmentationTimeout: number;
    fileTimeout: number;

    version?: boolean;
    verbose?: boolean;
    debug?: boolean;
    help?: boolean;
}

const runArgs = process.env["run-args"]?.split(" ") ?? [];

export const args = parse<BigRocqCliArguments>(
    {
        targetRootPath: { type: String, alias: "p", multiple: false },
        workspaceRootPath: { type: String, alias: "w", multiple: false },
        coqLspServerPath: {
            type: String,
            alias: "c",
            multiple: false,
            defaultValue: "coq-lsp",
        },
        theoremAugmentationTimeout: {
            type: Number,
            alias: "t",
            defaultValue: 300000,
            description:
                "Every produced sample is checked manually to be correctly compiling/type-checking. This timeout means how much time you give the util to check all new samples for a single theorem.",
        },
        fileTimeout: {
            type: Number,
            defaultValue: 90000,
            description:
                "Non-trivially explainable semantics. Set bigger if you have files with a lot of theorems or in case you fail with timeout.",
        },
        version: { type: Boolean, optional: true, alias: "v" },
        verbose: { type: Boolean, optional: true },
        debug: { type: Boolean, optional: true, alias: "d" },
        help: { type: Boolean, optional: true, alias: "h" },
    },
    {
        argv: runArgs,
        helpArg: "help",
        headerContentSections: [
            {
                header: "========================================",
            },
            {
                header: "BigRocq",
                content:
                    "This utility takes a Coq project as input and uses some domain knowladge to increase a number of theorems in the dataset by a significant factor.",
            },
        ],
        footerContentSections: [
            {
                content:
                    "Project home: {underline https://github.com/K-dizzled/coq-data-augmentation}",
            },
            {
                header: "========================================",
            },
        ],
    }
);
