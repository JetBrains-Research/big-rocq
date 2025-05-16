export type LogColor =
    | "red"
    | "green"
    | "yellow"
    | "blue"
    | "magenta"
    | "gray"
    | "cyan"
    | "reset";

export type LogWeight = "bold" | "reset";

export function print(
    text: string,
    color?: LogColor | undefined,
    weight?: LogWeight
) {
    console.log(style(colorize(text, color), weight));
}

export function colorize(text: string, color: LogColor | undefined): string {
    if (color === undefined) {
        return text;
    }
    const resetCode = colorCode("reset");
    const codeOfColor = colorCode(color);
    return `${codeOfColor}${text}${resetCode}`;
}

export function style(text: string, weight: LogWeight | undefined): string {
    if (weight === undefined) {
        return text;
    }
    const resetCode = weightCode("reset");
    const codeOfWight = weightCode(weight);
    return `${codeOfWight}${text}${resetCode}`;
}

export function colorCode(color: LogColor): string {
    switch (color) {
        case "reset":
            return "\x1b[0m";
        case "cyan":
            return "\x1b[36m";
        case "red":
            return "\x1b[31m";
        case "green":
            return "\x1b[32m";
        case "yellow":
            return "\x1b[33m";
        case "blue":
            return "\x1b[34m";
        case "magenta":
            return "\x1b[35m";
        case "gray":
            return "\x1b[90m";
    }
}

export function weightCode(color: LogWeight): string {
    switch (color) {
        case "reset":
            return "\x1b[0m";
        case "bold":
            return "\x1b[1m";
    }
}
