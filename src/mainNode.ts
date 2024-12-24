import { ProjectProcessor } from "./core/projectProcessor";

export async function run() {
    const projectProcessor = await ProjectProcessor.create();
    await projectProcessor.processProject();
}
