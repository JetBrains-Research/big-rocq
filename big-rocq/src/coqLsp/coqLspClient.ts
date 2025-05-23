import { Mutex } from "async-mutex";
import { readFileSync } from "fs";
import { Err, Ok, Result } from "ts-results";
import { OutputChannel } from "vscode";
import {
    BaseLanguageClient,
    Diagnostic,
    DidChangeTextDocumentNotification,
    DidChangeTextDocumentParams,
    DidCloseTextDocumentNotification,
    DidCloseTextDocumentParams,
    DidOpenTextDocumentNotification,
    DidOpenTextDocumentParams,
    Disposable,
    LogTraceNotification,
    Position,
    ProtocolNotificationType,
    PublishDiagnosticsNotification,
    RequestType,
    TextDocumentIdentifier,
    VersionedTextDocumentIdentifier,
} from "vscode-languageclient";

import { EventLogger } from "../logging/eventLogger";
import { Uri } from "../utils/uri";

import { CoqLspClientConfig, CoqLspServerConfig } from "./coqLspConfig";
import { CoqLspConnector } from "./coqLspConnector";
import {
    CoqLspError,
    CoqLspStartupError,
    FlecheDocument,
    FlecheDocumentParams,
    GoalAnswer,
    GoalConfig,
    GoalRequest,
    PpString,
} from "./coqLspTypes";

export interface DocumentSpec {
    uri: Uri;
    version?: number;
}

export interface CoqLspClient extends Disposable {
    /**
     * Fetches all goals present at the given position in the document.
     * This method doesn't open the document implicitly, therefore
     * it assumes that openTextDocument has been called before.
     * @param position Position in the document where to fetch the goals
     * @param documentUri Uri of the document
     * @param version Version of the document
     * @param command Optional command to execute before fetching the goals
     * @returns All goals present at the given position, not only the first one
     */
    getGoalsAtPoint(
        position: Position,
        documentUri: Uri,
        version: number,
        command?: string
    ): Promise<Result<GoalConfig<PpString>, Error>>;

    /**
     * Returns a FlecheDocument for the given uri.
     * This method doesn't open the document implicitly, therefore
     * it assumes that openTextDocument has been called before.
     */
    getFlecheDocument(uri: Uri): Promise<FlecheDocument>;

    updateTextDocument(
        oldDocumentText: string,
        appendedSuffix: string,
        uri: Uri,
        version: number,
        timeoutMillis?: number
    ): Promise<DiagnosticMessage>;

    openTextDocument(
        uri: Uri,
        version?: number,
        timeoutMillis?: number
    ): Promise<DiagnosticMessage>;

    closeTextDocument(uri: Uri): Promise<void>;

    withTextDocument<T>(
        documentSpec: DocumentSpec,
        block: (openedDocDiagnostic: DiagnosticMessage) => Promise<T>
    ): Promise<T>;
}

const goalReqType = new RequestType<GoalRequest, GoalAnswer<PpString>, void>(
    "proof/goals"
);

const flecheDocReqType = new RequestType<
    FlecheDocumentParams,
    FlecheDocument,
    void
>("coq/getDocument");

export type DiagnosticMessage = string | undefined;

export class CoqLspClientImpl implements CoqLspClient {
    private client: BaseLanguageClient;
    private subscriptions: Disposable[] = [];
    private mutex = new Mutex();

    private constructor(
        coqLspConnector: CoqLspConnector,
        public readonly eventLogger?: EventLogger,
        public readonly abortSignal?: AbortSignal
    ) {
        this.client = coqLspConnector;
    }

    static async create(
        serverConfig: CoqLspServerConfig,
        clientConfig: CoqLspClientConfig,
        logOutputChannel: OutputChannel,
        eventLogger?: EventLogger,
        abortSignal?: AbortSignal
    ): Promise<CoqLspClientImpl> {
        const connector = new CoqLspConnector(
            serverConfig,
            clientConfig,
            logOutputChannel
        );
        await connector.start().catch((error) => {
            throw new CoqLspStartupError(
                `failed to start coq-lsp with Error: ${error.message}`,
                clientConfig.coq_lsp_server_path
            );
        });
        return new CoqLspClientImpl(connector, eventLogger, abortSignal);
    }

    async getGoalsAtPoint(
        position: Position,
        documentUri: Uri,
        version: number,
        command?: string
    ): Promise<Result<GoalConfig<PpString>, Error>> {
        return await this.mutex.runExclusive(async () => {
            return this.getGoalsAtPointUnsafe(
                position,
                documentUri,
                version,
                command
            );
        });
    }

    async updateTextDocument(
        oldDocumentText: string,
        appendedSuffix: string,
        uri: Uri,
        version: number = 1,
        timeoutMillis: number = 900000
    ): Promise<DiagnosticMessage> {
        return await this.mutex.runExclusive(async () => {
            return this.updateTextDocumentUnsafe(
                oldDocumentText,
                appendedSuffix,
                uri,
                version,
                timeoutMillis
            );
        });
    }

    async openTextDocument(
        uri: Uri,
        version: number = 1,
        timeoutMillis: number = 900000
    ): Promise<DiagnosticMessage> {
        return await this.mutex.runExclusive(async () => {
            return this.openTextDocumentUnsafe(uri, version, timeoutMillis);
        });
    }

    async closeTextDocument(uri: Uri): Promise<void> {
        return await this.mutex.runExclusive(async () => {
            return this.closeTextDocumentUnsafe(uri);
        });
    }

    async withTextDocument<T>(
        documentSpec: DocumentSpec,
        block: (openedDocDiagnostic: DiagnosticMessage) => Promise<T>
    ): Promise<T> {
        const diagnostic = await this.openTextDocument(
            documentSpec.uri,
            documentSpec.version
        );
        // TODO: discuss whether coq-lsp is capable of maintaining several docs opened
        // or we need a common lock for open-close here
        try {
            return await block(diagnostic);
        } finally {
            await this.closeTextDocument(documentSpec.uri);
        }
    }

    async getFlecheDocument(uri: Uri): Promise<FlecheDocument> {
        return await this.mutex.runExclusive(async () => {
            return this.getFlecheDocumentUnsafe(uri);
        });
    }

    /**
     * Dirty due to the fact that the client sends no pure
     * error: https://github.com/ejgallego/coq-lsp/blob/f98b65344c961d1aad1e0c3785199258f21c3abc/controller/request.ml#L29
     */
    cleanLspError(errorMsg?: string): string | undefined {
        const errorMsgPrefixRegex = /^Error in .* request: (.*)$/s;
        if (!errorMsg) {
            return undefined;
        }
        const match = errorMsg.match(errorMsgPrefixRegex);
        return match ? match[1] : undefined;
    }

    removeTraceFromLspError(errorMsgWithTrace: string): string | undefined {
        const traceStartString = "Raised at";

        if (!errorMsgWithTrace.includes(traceStartString)) {
            return errorMsgWithTrace.split("\n").shift();
        }

        return errorMsgWithTrace
            .substring(0, errorMsgWithTrace.indexOf(traceStartString))
            .trim();
    }

    filterDiagnostics(
        diagnostics: Diagnostic[],
        position: Position
    ): string | undefined {
        const diagnosticMessageWithTrace = diagnostics
            .filter((diag) => diag.range.start.line >= position.line)
            .filter((diag) => diag.severity === 1) // 1 is error
            .shift()?.message;

        if (!diagnosticMessageWithTrace) {
            return undefined;
        } else {
            return this.removeTraceFromLspError(diagnosticMessageWithTrace);
        }
    }

    private async getGoalsAtPointUnsafe(
        position: Position,
        documentUri: Uri,
        version: number,
        command?: string
    ): Promise<Result<GoalConfig<PpString>, Error>> {
        let goalRequestParams: GoalRequest = {
            textDocument: VersionedTextDocumentIdentifier.create(
                documentUri.uri,
                version
            ),
            position,
            // eslint-disable-next-line @typescript-eslint/naming-convention
            pp_format: "Str",
            command: command,
        };

        try {
            const goalAnswer = await this.client.sendRequest(
                goalReqType,
                goalRequestParams
            );
            const goals = goalAnswer?.goals;

            if (!goals) {
                return Err(CoqLspError.unknownError());
            }

            return Ok(goals);
        } catch (e) {
            if (e instanceof Error) {
                const errorMsg = this.cleanLspError(
                    this.removeTraceFromLspError(e.message)
                );
                if (errorMsg) {
                    return Err(new CoqLspError(errorMsg));
                }
                return Err(
                    new CoqLspError(
                        "Unable to parse CoqLSP error, please report this issue: " +
                            e.message
                    )
                );
            }

            return Err(CoqLspError.unknownError());
        }
    }

    private sleep(ms: number): Promise<ReturnType<typeof setTimeout>> {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    private async waitUntilFileFullyChecked(
        requestType: ProtocolNotificationType<any, any>,
        params: any,
        uri: Uri,
        version: number,
        lastDocumentEndPosition?: Position,
        timeoutMillis: number = 900000
    ): Promise<DiagnosticMessage> {
        await this.client.sendNotification(requestType, params);

        let pendingProgress = true;
        let pendingDiagnostic = true;
        let awaitedDiagnostics: Diagnostic[] | undefined = undefined;

        this.subscriptions.push(
            this.client.onNotification(LogTraceNotification.type, (params) => {
                if (params.message.includes("document fully checked")) {
                    pendingProgress = false;
                }
            })
        );

        this.subscriptions.push(
            this.client.onNotification(
                PublishDiagnosticsNotification.type,
                (params) => {
                    if (
                        params.uri.toString() === uri.uri &&
                        params.version === version
                    ) {
                        pendingDiagnostic = false;
                        awaitedDiagnostics = params.diagnostics;

                        if (
                            lastDocumentEndPosition &&
                            this.filterDiagnostics(
                                params.diagnostics,
                                lastDocumentEndPosition
                            ) !== undefined
                        ) {
                            pendingProgress = false;
                        }
                    }
                }
            )
        );

        while (timeoutMillis > 0 && (pendingProgress || pendingDiagnostic)) {
            await this.sleep(100);
            timeoutMillis -= 100;
        }

        if (
            timeoutMillis <= 0 ||
            pendingProgress ||
            pendingDiagnostic ||
            awaitedDiagnostics === undefined
        ) {
            throw new CoqLspError("coq-lsp did not respond in time");
        }

        this.subscriptions.forEach((d) => d.dispose());

        return this.filterDiagnostics(
            awaitedDiagnostics,
            lastDocumentEndPosition ?? Position.create(0, 0)
        );
    }

    private getTextEndPosition(lines: string[]): Position {
        return Position.create(
            lines.length - 1,
            lines[lines.length - 1].length
        );
    }

    private async updateTextDocumentUnsafe(
        oldDocumentText: string,
        appendedSuffix: string,
        uri: Uri,
        version: number = 1,
        timeoutMillis: number = 900000
    ): Promise<DiagnosticMessage> {
        const updatedText = oldDocumentText + appendedSuffix;
        const oldEndPosition = this.getTextEndPosition(
            oldDocumentText.split("\n")
        );

        const params: DidChangeTextDocumentParams = {
            textDocument: {
                uri: uri.uri,
                version: version,
            },
            contentChanges: [
                {
                    text: updatedText,
                },
            ],
        };

        return await this.waitUntilFileFullyChecked(
            DidChangeTextDocumentNotification.type,
            params,
            uri,
            version,
            oldEndPosition,
            timeoutMillis
        );
    }

    private async openTextDocumentUnsafe(
        uri: Uri,
        version: number = 1,
        timeoutMillis: number = 900000
    ): Promise<DiagnosticMessage> {
        const docText = readFileSync(uri.fsPath).toString();

        const params: DidOpenTextDocumentParams = {
            textDocument: {
                uri: uri.uri,
                languageId: "coq",
                version: version,
                text: docText,
            },
        };

        return await this.waitUntilFileFullyChecked(
            DidOpenTextDocumentNotification.type,
            params,
            uri,
            version,
            undefined,
            timeoutMillis
        );
    }

    private async closeTextDocumentUnsafe(uri: Uri): Promise<void> {
        const params: DidCloseTextDocumentParams = {
            textDocument: {
                uri: uri.uri,
            },
        };

        await this.client.sendNotification(
            DidCloseTextDocumentNotification.type,
            params
        );
    }

    private async getFlecheDocumentUnsafe(uri: Uri): Promise<FlecheDocument> {
        let textDocument = TextDocumentIdentifier.create(uri.uri);
        let params: FlecheDocumentParams = { textDocument };
        const doc = await this.client.sendRequest(flecheDocReqType, params);

        return doc;
    }

    /**
     * _Implementation note:_ although this `dispose` implementation calls an async method,
     * we are not tend to await it, as well as `CoqLspClient.dispose()` in general.
     *
     * Since different coq-lsp clients correspond to different processes,
     * they don't have any shared resources; therefore, a new client can be started without
     * waiting for the previous one to finish. Thus, we don't await for this `dispose()` call
     * to finish, the client and its process will be disposed at some point asynchronously.
     */
    dispose(): void {
        this.subscriptions.forEach((d) => d.dispose());
        this.client.stop();
    }
}
