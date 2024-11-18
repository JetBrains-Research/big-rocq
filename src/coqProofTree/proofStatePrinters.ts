/* eslint-disable @typescript-eslint/naming-convention */
import { Goal, GoalConfig, Hyp, PpString } from "../coqLsp/coqLspTypes";
import { colorize, LogColor, LogWeight, print, style } from "../logging/colorLogging";

type PpMode = "console" | "html";

const htmlColors = {
    bold: (text: string) => `<b>${text}</b>`,
    red: (text: string) => `<span style="color: red;">${text}</span>`,
    green: (text: string) => `<span style="color: green;">${text}</span>`,
    yellow: (text: string) => `<span style="color: goldenrod;">${text}</span>`,
    blue: (text: string) => `<span style="color: blue;">${text}</span>`,
    cyan: (text: string) => `<span style="color: cyan;">${text}</span>`,
};

function colorizeText(text: string, ppMode: PpMode, color?: LogColor): string {
    if (ppMode === "console") {
        return colorize(text, color);
    } else {
        switch (color) {
            case "red": return htmlColors.red(text);
            case "green": return htmlColors.green(text);
            case "yellow": return htmlColors.yellow(text);
            case "blue": return htmlColors.blue(text);
            case "cyan": return htmlColors.cyan(text);
            default: return text;
        }
    }
}

function styleText(text: string, ppMode: PpMode, styleType?: LogWeight): string {
    if (ppMode === "console") {
        return style(text, styleType);
    } else {
        if (styleType === "bold") return htmlColors.bold(text);
        return text;
    }
}

export function ppHypothesis(hyp: Hyp<PpString>, ppMode: PpMode): string {
    const names = colorizeText(hyp.names.join(", "), ppMode, "yellow");
    const type = colorizeText(hyp.ty as string, ppMode, "blue");
    const def = hyp.def ? ` := ${colorizeText(hyp.def as string, ppMode, "green")}` : "";
    return `${names}${def} : ${type}`;
}

export function ppGoal(goal: Goal<PpString>, ppMode: PpMode): string {
    const delimiter = ppMode === "console" ? '\n' : '<br>';
    const hyps = goal.hyps.map(hyp => ppHypothesis(hyp, ppMode)).join(delimiter);
    const conclusion = colorizeText(goal.ty as string, ppMode, "red");
    const middleBreak = hyps.length === 0 ? '' : delimiter;
    return `${hyps}${middleBreak}${styleText(conclusion, ppMode, "bold")}\n`;
}

export function printGoalStack(stack: [Goal<PpString>[], Goal<PpString>[]][], ppMode: PpMode) {
    const stackDepth = stack.length;
    let currentDepth = 0;

    while (currentDepth < stackDepth) {
        const [l, r] = stack[currentDepth];
        const goals = l.concat(r);

        if (goals.length > 0) {
            let levelIndicator = currentDepth === 0
                ? "the same bullet level"
                : `the -${currentDepth} bullet level`;
            print(colorizeText(`Remaining goals at ${levelIndicator}`, ppMode, "cyan"));

            goals.forEach((goal, index) => {
                print(styleText(`Goal (${index + 1})`, ppMode, "bold"));
                print(ppGoal(goal, ppMode));
            });
        }

        currentDepth++;
    }
}

export function printGoals(goals: Goal<PpString>[], ppMode: PpMode) {
    print(styleText(`Goals (${goals.length})`, ppMode, "bold"));
    goals.forEach((goal, index) => {
        print(styleText(`Goal (${index + 1})`, ppMode, "bold"));
        print(ppGoal(goal, ppMode));
    });
}

export function printProofState(proofState: GoalConfig<PpString>, ppMode: PpMode) {
    const { goals, stack } = proofState;
    printGoals(goals, ppMode);
    printGoalStack(stack, ppMode);
}
