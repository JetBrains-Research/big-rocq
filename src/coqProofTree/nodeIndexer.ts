// TODO: Awful architectural decisions, refactor ProofTreeNode to make reasonable
// Issue is that I dont want indeces for nodes to be nullable.
export class NodeIndexer {
    private currentIndex = 0;

    get next(): number {
        return this.currentIndex++;
    }
}