import { CoqDatasetStats } from "../../coqDatasetModels";
import {
    DirectoryItemView,
    redirectToDirItem,
} from "../../generateDatasetViewer";

export const folderViewHtml = (
    dirItems: DirectoryItemView[],
    aggregatedStats: CoqDatasetStats,
    canGoBack: boolean
) => {
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Folder View</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #555;
            font-size: 1.8rem;
        }
        .stats-summary {
            background: linear-gradient(to right, #e3f2fd, #e8f5e9);
            border: 1px solid #cfd8dc;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 6px;
            font-size: 1em;
            color: #37474f;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .stats-summary .stat {
            flex: 1;
            text-align: center;
            font-weight: bold;
        }
        .stats-summary .stat span {
            display: block;
            font-size: 1.2em;
            color: #1e88e5;
        }
        ul {
            list-style: none;
            padding: 0;
        }
        li {
            padding: 12px;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background-color 0.3s;
        }
        li:last-child {
            border-bottom: none;
        }
        li:hover {
            background-color: #f9f9f9;
        }
        .icon {
            margin-right: 10px;
        }
        a {
            text-decoration: none;
            color: #007bff;
            font-weight: bold;
            transition: color 0.3s;
        }
        a:hover {
            color: #0056b3;
        }
        .folder {
            color: #f39c12;
        }
        .file {
            color: #2980b9;
        }
        .back-button {
            position: absolute;
            top: 10px;
            left: 10px;
            padding: 10px 15px;
            background: linear-gradient(to right, #1980f6, #00c6ff);
            text-decoration: none;
            color: white;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 500;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .back-button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
        }
        .stats {
            font-size: 0.9em;
            color: #777;
            margin-left: auto;
        }
    </style>
</head>
<body>
    ${canGoBack ? `<a href="../index_dir_view.html" class="back-button">🔙</a>` : ""}
    <div class="container">
        <h1>Folder View</h1>
        <div class="stats-summary">
            <div class="stat">
                Total Items<br><span>${dirItems.length}</span>
            </div>
            <div class="stat">
                Successful theorems<br><span>${aggregatedStats.proofTreeBuildRatio[0]} / ${aggregatedStats.proofTreeBuildRatio[1]}</span>
            </div>
            <div class="stat">
                Augmented Nodes<br><span>${aggregatedStats.augmentedNodesRatio[0]} / ${aggregatedStats.augmentedNodesRatio[1]}</span>
            </div>
        </div>
        <ul>
            ${dirItems
                .map(
                    (item) => `
                <li>
                    <a class="${item.isDir ? "folder" : "file"}" href="${redirectToDirItem(item)}" target="_self">
                        <span class="icon">${item.isDir ? "📁" : "📄"}</span>
                        ${item.name}
                    </a>
                    <span class="stats">(${item.itemStats.augmentedNodesRatio[0]} / ${item.itemStats.augmentedNodesRatio[1]} nodes augmented)</span>
                </li>`
                )
                .join("")}
        </ul>
    </div>
</body>
</html>`;
};
