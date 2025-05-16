/* eslint-disable @typescript-eslint/naming-convention */
export class GoalSelector {
    protected constructor() {}

    static AllGoals(): GoalSelector {
        return new AllGoals();
    }

    static SingleGoal(n: number): GoalSelector {
        return new SingleGoal(n);
    }

    static GoalList(indices: number[]): GoalSelector {
        return new GoalList(indices);
    }
}

export class AllGoals extends GoalSelector {
    static readonly type = "SelectAll";
}

export class SingleGoal extends GoalSelector {
    static readonly type = "SelectNth";
    constructor(public readonly n: number) {
        super();
    }
}

export class GoalList extends GoalSelector {
    static readonly type = "SelectList";
    constructor(public readonly indices: number[]) {
        super();
    }
}
