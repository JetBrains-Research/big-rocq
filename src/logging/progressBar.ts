import { create } from "ts-progress";

export function getProgressBar(fileName: string, total: number) {
    return create({
        total: total,
        pattern: `Progress on file ${fileName}: {bar.white.blue.20} {current}/{total} | Remaining: {remaining} | Elapsed: {elapsed} `,
        textColor: "blue",
        updateFrequency: 300,
    });
}
