export async function measureElapsedMillis<T>(
    block: () => Promise<T>
): Promise<[T, number]> {
    const timeMark = new TimeMark();
    const result = await block();
    return [result, timeMark.measureElapsedMillis()];
}

export class TimeMark {
    private startTime: number;

    constructor() {
        this.startTime = performance.now();
    }

    measureElapsedMillis(): number {
        return Math.round(performance.now() - this.startTime);
    }
}