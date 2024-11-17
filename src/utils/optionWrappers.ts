export function unwrapOrThrow<T>(value: T | undefined): T {
    if (!value) {
        throw new Error("Unable to unwrap a value of undefined");
    }

    return value;
}
