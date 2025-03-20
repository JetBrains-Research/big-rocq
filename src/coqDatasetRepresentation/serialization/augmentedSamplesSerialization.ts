export interface CompactSerializedAugmentedSample {
    statement: string;
    proofString: string;
}

export interface CompactSerializedAugmentedSamples {
    fileSamples: CompactSerializedAugmentedSample[];
    filePath: string;
}
