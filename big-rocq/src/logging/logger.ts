import pino from "pino";

import { EventLogger, Severity, anyEventKeyword } from "../logging/eventLogger";

class Logger {
    private readonly outputStreamWriter = pino({
        formatters: {
            level: (label) => {
                return { level: label };
            },
        },
        level: process.env.LOG_LEVEL || "info",
        redact: {
            paths: ["*.password", "*.token", "*.apiKey"],
            censor: "***censored***",
        },
    });

    constructor(eventLogger: EventLogger, logLevel: Severity = Severity.INFO) {
        eventLogger.subscribe(anyEventKeyword, logLevel, (message, data) => {
            this.outputStreamWriter.info({
                message,
                data,
            });
        });
    }

    dispose(): void {
        this.outputStreamWriter.flush();
    }
}

export default Logger;
