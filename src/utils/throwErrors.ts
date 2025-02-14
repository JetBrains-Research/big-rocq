export function illegalState(...message: string[]): never {
    throw new IllegalStateError(`Illegal state: ${message.join("")}`);
}

export class IllegalStateError extends Error {
    constructor(message: string) {
        super(message);
        Object.setPrototypeOf(this, new.target.prototype);
        this.name = "IllegalStateError";
    }
}
