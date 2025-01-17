import { parseUtilityRunParams } from "./cli/argParser";
import { BigRocqCliArguments, args } from "./cli/args";
import { ProjectProcessor } from "./core/projectProcessor";
import { print } from "./logging/colorLogging";

export async function run() {
    const runArgs = args;
    const terminate = await preprocessCommands(runArgs);

    if (terminate) {
        return;
    }

    const validatedRunArgs = parseUtilityRunParams(runArgs);
    const projectProcessor = await ProjectProcessor.create(validatedRunArgs);
    await projectProcessor.processProject();
}

/**
 * As of now, only two commands apart from the default command are supported:
 * -help -- covered by ts-command-line-args
 * -version
 */
async function preprocessCommands(args: BigRocqCliArguments): Promise<boolean> {
    if (args.version) {
        const packageJson = await import("../package.json");
        const versionData = packageJson.default || packageJson;

        if (!versionData || !versionData.version) {
            console.error("Could not find the version data");
        } else {
            print(
                ` 
            ======================
            BigRocq version: ${versionData.version}
            ======================
            `,
            "yellow",
            "bold"
            );
        }

        return true;
    }

    return false;
}
