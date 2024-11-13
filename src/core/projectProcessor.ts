import { createCoqLspClient } from "../coqLsp/coqLspBuilders";
import { CoqLspClient } from "../coqLsp/coqLspClient";

import { parseCoqFile } from "../coqParser/parseCoqFile";
import { EventLogger } from "../logging/eventLogger";
import Logger from "../logging/logger";
import { Uri } from "../utils/uri";

import { defaultUtilityRunParams } from "./utilityRunParams";

export class ProjectProcessor {
    private constructor(
        public readonly eventLogger: EventLogger,
        public readonly logger: Logger,
        public readonly abortController: AbortController,
        private readonly coqLspClient: CoqLspClient
    ) {}

    static async create(): Promise<ProjectProcessor> {
        const eventLogger: EventLogger = new EventLogger();
        const logger: Logger = new Logger(
            eventLogger,
            defaultUtilityRunParams.loggingLevel
        );
        const abortController = new AbortController();

        const coqLspClient = await createCoqLspClient(
            defaultUtilityRunParams.coqLspServerPath,
            undefined,
            eventLogger,
            abortController
        );

        return new ProjectProcessor(
            eventLogger,
            logger,
            abortController,
            coqLspClient
        );
    }

    // async run(filePath: string): Promise<void> {
    //     await this.processFile(filePath);
    // }

    async processFile(filePath: string): Promise<void> {
        this.eventLogger.log(
            "started-processing-file",
            `Processing file ${filePath}`
        );

        const fileUri = Uri.fromPath(filePath);
        await this.coqLspClient.openTextDocument(fileUri);

        const parsedTheorems = await parseCoqFile(
            fileUri,
            this.coqLspClient,
            this.abortController.signal,
            false,
            this.eventLogger
        );

        this.eventLogger.log(
            "finished-processing-file",
            `Retrieved following theorems from file ${filePath}: ${parsedTheorems
                .map((theorem) => theorem.name)
                .join(", ")}`
        );

        const firstTheorem = parsedTheorems[0];
        
        

        await this.coqLspClient.closeTextDocument(fileUri);
    }

    dispose(): void {
        this.logger.dispose();
    }
}
