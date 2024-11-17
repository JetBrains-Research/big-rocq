import { ProjectProcessor } from "./core/projectProcessor";

export async function run() {
    // TODO: Pass root of project, scan for all .v files, and process them
    const filePath = "dataset/standalone-source-files/auto_benchmark.v";
    const projectProcessor = await ProjectProcessor.create();
    await projectProcessor.processFile(filePath);
}

// (async () => {
//     await run();
// })();
