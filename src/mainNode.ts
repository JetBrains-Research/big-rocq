// import { generateFileViewer } from "./coqDatasetRepresentation/generateDatasetViewer";
import { ProjectProcessor } from "./core/projectProcessor";

export async function run() {
    // TODO: Pass root of project, scan for all .v files, and process them
    const filePath = "dataset/standalone-source-files";
    const projectProcessor = await ProjectProcessor.create();
    await projectProcessor.processProject(filePath);
    // const coqDataset = [fileItem];
    // generateFileViewer(coqDataset[0]);
}
