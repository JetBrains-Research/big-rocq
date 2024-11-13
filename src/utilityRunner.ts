import { runTests } from "@vscode/test-electron";
import * as path from "path";

async function main() {
    try {
        const extensionDevelopmentPath = path.resolve(__dirname, "../../");
        const extensionTestsPath = path.resolve(__dirname, "./mainNode");

        const args = process.argv.slice(2);
        const envVals = args.reduce(
            (acc, arg) => {
                const [key, val] = arg.split("=");
                acc[`TEST_ARG${key}`] = val;
                return acc;
            },
            {} as { [key: string]: string | undefined }
        );

        await runTests({
            extensionDevelopmentPath,
            extensionTestsPath,
            extensionTestsEnv: envVals,
        });
    } catch (err) {
        console.error("Failed to run the utility", err);
        process.exit(1);
    }
}

main();
