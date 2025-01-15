import { Result } from "ts-results";

export interface TheoremList {
    thrName: string;
    augmentedNodesRatio?: [number, number];
    filePath: Result<string, Error>;
}

// TODO: Remove `back-button` when only one file is target
export const theoremListViewHtml = (theoremList: TheoremList[]) => `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Theorem List</title>
    <link href='https://fonts.googleapis.com/css?family=Fira Code' rel='stylesheet'>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        h1 {
            text-align: center;
            color: #555;
            font-size: 1.8rem;
        }
        .container {
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
        }
        ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        li {
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
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
        .success {
            color: #44cf6c;
        }
        .thr-name {
            font-family: 'Fira Code', monospace;
            font-weight: bold;
        }
        .error {
            color: #c6212c;
        }
        a {
            text-decoration: none;
            color: #2980b9;
            font-weight: bold;
            transition: color 0.3s;
        }
        a:hover {
            color: #0056b3;
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        .tooltip .tooltip-text {
            visibility: hidden;
            width: 200px;
            background-color: #333;
            color: #fff;
            font-weight: bold;
            text-align: center;
            padding: 5px;
            border-radius: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
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
            margin-right: 10px;
            font-size: 0.9em;
            color: #777;
        }
    </style>
</head>
<body>
    <a href="../index_dir_view.html" class="back-button">🔙</a>
    <div class="container">
        <h1>Theorem List</h1>
        <ul>
            ${theoremList
                .map(({ thrName, filePath, augmentedNodesRatio }) => {
                    const isSuccess = filePath.ok;
                    const linkOrMessage = isSuccess
                        ? `<a href="${filePath.val}" target="_self">View Theorem</a>`
                        : `<span class="tooltip error">Unable to augment theorem
                            <span class="tooltip-text">${filePath.val.message}</span>
                        </span>`;
                    const stats = augmentedNodesRatio
                        ? `<span class="stats">(${augmentedNodesRatio[0]} / ${augmentedNodesRatio[1]} nodes augmented)</span>`
                        : "";
                    return `
                        <li class="${isSuccess ? "success" : "error"}">
                            <span class="thr-name">${thrName}</span>
                            ${isSuccess ? stats : ""}
                            ${linkOrMessage}
                        </li>`;
                })
                .join("")}
        </ul>
    </div>
</body>
</html>`;
