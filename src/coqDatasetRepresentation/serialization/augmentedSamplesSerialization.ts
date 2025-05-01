export interface CompactSerializedAugmentedSample {
    statement: string;
    conclusion: string;
    hypotheses: string;
    proofString: string;
}

export interface CompactSerializedAugmentedSamples {
    fileSamples: CompactSerializedAugmentedSample[];
    filePath: string;
}
