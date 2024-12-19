import { DirectoryItem, redirectToDirItem } from "../../generateDatasetViewer";

export const folderViewHtml = (dirItems: DirectoryItem[], canGoBack: boolean) => `
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
    </style>
</head>
<body>
    ${canGoBack ? `<a href="../index_dir_view.html" class="back-button">🔙</a>` : ""}
    <div class="container">
        <h1>Folder View</h1>
        <ul>
            ${dirItems
                .map((item) => `
                <li>
                    <a class="${item.isDir ? "folder" : "file"}" href="${redirectToDirItem(item)}" target="_self">
                        <span class="icon">${item.isDir ? '📁' : '📄'}</span>
                        ${item.name}
                    </a>
                </li>`)
                .join("")}
        </ul>
    </div>
</body>
</html>`;
