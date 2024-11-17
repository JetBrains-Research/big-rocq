// export interface TreeNodeInterface<EdgeType> {
//     parent: TreeNodeInterface<EdgeType> | null;
//     children: [TreeNodeInterface<EdgeType>, EdgeType][];
// }

// export class TreeNode<NodeType, EdgeType> {
//     parent: NodeType | undefined;
//     children: [NodeType, EdgeType][] = [];

//     protected constructor(
//         parent?: [NodeType, EdgeType],
//     ) {
//         if (!parent) {
//             this.parent = undefined;
//             return;
//         }

//         const [parentNode, stepLabel] = parent;
//         this.parent = parentNode;
//         this.parent.children.push([this, stepLabel]);
//     }
// }
