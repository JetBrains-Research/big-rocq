import * as fs from "fs";
import { Position } from "vscode-languageclient";

export function ensureDirExists(dir: string): void {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
}

export function getTextInRange(
    start: Position,
    end: Position,
    lines: string[],
    preserveLineBreaks = false
): string {
    if (start.line === end.line) {
        return lines[start.line].substring(start.character, end.character);
    } else {
        let text = lines[start.line].substring(start.character);
        for (let i = start.line + 1; i < end.line; i++) {
            if (preserveLineBreaks) {
                text += "\n";
            }
            text += lines[i];
        }
        if (preserveLineBreaks) {
            text += "\n";
        }
        text += lines[end.line].substring(0, end.character);

        return text;
    }
}
