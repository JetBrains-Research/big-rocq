import * as fs from "fs";
import * as path from "path";
import { Err, Ok } from "ts-results";

import { CoqDatasetAugmentedFile } from "../coqDatasetRepresentation/coqDatasetModels";

import { proofTreeViewHtml } from "./datasetVisualization/templates/proofTree";
import {
    TheoremList,
    theoremListViewHtml,
} from "./datasetVisualization/templates/theoremList";
import {
    folderViewHtml
} from "./datasetVisualization/templates/dirView";
import { compactSerializeCoqTheorem } from "./serialization/coqDatasetCompactSerialization";

// TODO: Move to params
export const coqDataViewDir = "tempDatasetView";

function ensureDirExists(dir: string): void {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
}

export interface DirectoryItem {
    name: string; 
    isDir: boolean;
    isFile: boolean;
}

function dirItemFrom(fsDirent: fs.Dirent): DirectoryItem {
    return {
        name: path.parse(fsDirent.name).name,
        isDir: fsDirent.isDirectory(),
        isFile: fsDirent.isFile(),
    };
}

// TODO: Move to params
export const fileIndexHtmlName = "index_file_view";
export const dirIndexHtmlName = "index_dir_view";
export function redirectToDirItem(dirItem: DirectoryItem): string {
    return dirItem.isDir ? dirItem.name + `/${dirIndexHtmlName}.html` : dirItem.name + `/${fileIndexHtmlName}.html`;
}


export function generateFolderViewer(
    rootPath: string,
    curDirPath: string,
    items: fs.Dirent[],
) {
    ensureDirExists(coqDataViewDir);
    const relativeFromRoot = path.relative(rootPath, curDirPath);
    const workingDir = path.join(coqDataViewDir, relativeFromRoot);
    ensureDirExists(workingDir);

    const dirIndexHtmlPath = path.join(workingDir, `${dirIndexHtmlName}.html`);
    const folderViewItems = items.map(dirItemFrom);
    const htmlTemplate = folderViewHtml(folderViewItems, rootPath != curDirPath);

    fs.writeFileSync(dirIndexHtmlPath, htmlTemplate);
}

export function generateFileViewer(
    rootPath: string,
    coqDatasetFile: CoqDatasetAugmentedFile
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
            thr_name: augmentedTheorem.parsedTheorem.name,
            file_path: augmentedTheorem.theoremAugmentationRes.ok
                ? Ok(`${augmentedTheorem.parsedTheorem.name}.html`)
                : Err(
                        Error(
                            augmentedTheorem.theoremAugmentationRes.val
                                .message
                        )
                    ),
        })
    );
    const htmlTemplate = theoremListViewHtml(theoremList);
    fs.writeFileSync(fileIndexHtmlPath, htmlTemplate);

    coqDatasetFile.augmentedTheorems.forEach((augmentedTheorem) => {
        const theoremName = augmentedTheorem.parsedTheorem.name;
        const compactSerializedTheorem = compactSerializeCoqTheorem(
            augmentedTheorem,
            "html"
        );
        
        const htmlFilePath = path.join(fileDir, `${theoremName}.html`);
        const htmlTemplate = proofTreeViewHtml(
            compactSerializedTheorem
        );
        fs.writeFileSync(htmlFilePath, htmlTemplate);
    });
}
