import * as fs from "fs";
import * as path from "path";
import { Err, Ok } from "ts-results";

import {
    CoqDatasetAugmentedFile,
    CoqDatasetDirItem,
    CoqDatasetFolder,
    CoqDatasetStats,
} from "../coqDatasetRepresentation/coqDatasetModels";

import { folderViewHtml } from "./datasetVisualization/templates/dirView";
import { proofTreeViewHtml } from "./datasetVisualization/templates/proofTree";
import {
    TheoremList,
    theoremListViewHtml,
} from "./datasetVisualization/templates/theoremList";
import { generateCoqAugmentedFile } from "./generateCoqAugmentedFile";
import { compactSerializeCoqTheorem } from "./serialization/coqDatasetCompactSerialization";

// TODO: Move to params
export const coqDataViewDir = "tempDatasetView";

function ensureDirExists(dir: string): void {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
}

export interface DirectoryItemView {
    name: string;
    isDir: boolean;
    isFile: boolean;
    itemStats: CoqDatasetStats;
}

// TODO: Move to params
export const fileIndexHtmlName = "index_file_view";
export const dirIndexHtmlName = "index_dir_view";
export function redirectToDirItem(dirItem: DirectoryItemView): string {
    return dirItem.isDir
        ? dirItem.name + `/${dirIndexHtmlName}.html`
        : dirItem.name + `/${fileIndexHtmlName}.html`;
}

export function generateFolderViewer(
    rootPath: string,
    curDirPath: string,
    datasetFolder: CoqDatasetFolder
) {
    ensureDirExists(coqDataViewDir);
    const relativeFromRoot = path.relative(rootPath, curDirPath);
    const workingDir = path.join(coqDataViewDir, relativeFromRoot);
    ensureDirExists(workingDir);

    const items = datasetFolder.dirItems.map(viewFromDatasetItem);

    const dirIndexHtmlPath = path.join(workingDir, `${dirIndexHtmlName}.html`);
    const htmlTemplate = folderViewHtml(
        items,
        datasetFolder.stats,
        rootPath !== curDirPath
    );

    fs.writeFileSync(dirIndexHtmlPath, htmlTemplate);
}

function viewFromDatasetItem(dirItem: CoqDatasetDirItem): DirectoryItemView {
    return {
        name: dirItem.itemName,
        isDir: dirItem.type === "dir",
        isFile: dirItem.type === "file",
        itemStats: dirItem.stats,
    };
}

export function generateFileViewer(
    rootPath: string,
    coqDatasetFile: CoqDatasetAugmentedFile,
    generateAugmentedCoqFile: boolean
): void {
    ensureDirExists(coqDataViewDir);

    const relativeFromRoot = path.relative(rootPath, coqDatasetFile.filePath);

    const fileSubdir = path.dirname(relativeFromRoot);
    const fileName = path.parse(relativeFromRoot).name;
    const fileDir = path.join(coqDataViewDir, fileSubdir, fileName);
    ensureDirExists(fileDir);

    const fileIndexHtmlPath = path.join(fileDir, `${fileIndexHtmlName}.html`);
    const theoremList: TheoremList[] = coqDatasetFile.augmentedTheorems.map(
        (augmentedTheorem) => ({
            thrName: augmentedTheorem.parsedTheorem.name,
            filePath: augmentedTheorem.proofTreeBuildResult.ok
                ? Ok(`${augmentedTheorem.parsedTheorem.name}.html`)
                : Err(Error(augmentedTheorem.proofTreeBuildResult.val.message)),
            augmentedNodesRatio: augmentedTheorem.stats.augmentedNodesRatio,
        })
    );
    const htmlTemplate = theoremListViewHtml(theoremList);
    fs.writeFileSync(fileIndexHtmlPath, htmlTemplate);
    if (generateAugmentedCoqFile) {
        generateCoqAugmentedFile(fileDir, coqDatasetFile);
    }

    coqDatasetFile.augmentedTheorems.forEach((augmentedTheorem) => {
        const theoremName = augmentedTheorem.parsedTheorem.name;
        const compactSerializedTheorem = compactSerializeCoqTheorem(
            augmentedTheorem,
            "html"
        );

        const htmlFilePath = path.join(fileDir, `${theoremName}.html`);
        const htmlTemplate = proofTreeViewHtml(compactSerializedTheorem);
        fs.writeFileSync(htmlFilePath, htmlTemplate);
    });
}
