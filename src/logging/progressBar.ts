import { create } from "ts-progress";

export function getProgressBar(processName: string, fileName: string, total: number) {
    return create({
        total: total,
        pattern: `${processName} file ${fileName}: {bar.white.blue.20} {current}/{total} | Remaining: {remaining} | Elapsed: {elapsed} `,
        textColor: "blue",
        updateFrequency: 300,
    });
}
