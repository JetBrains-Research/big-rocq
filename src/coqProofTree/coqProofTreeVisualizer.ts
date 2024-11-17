import * as fs from 'fs';
import { CoqProofTreeNode } from './coqProofTreeNode';
import { ppGoal } from './proofStatePrinters';

export class TreeVisualizer {
    constructor(private treeRoot: CoqProofTreeNode) {}

    // Serialize the proof tree to a structure compatible with D3
    serializeNode(treeNode: CoqProofTreeNode): any {
        return {
            proofState: treeNode.proofState ? ppGoal(treeNode.proofState, "html") : "Root",
            children: treeNode.children.map(([node, edge]) => ({
                node: this.serializeNode(node),
                edge: edge.text
            }))
        };
    }

    // Draw the proof tree to an HTML file
    drawToFile(filePath: string) {
        const serializedRoot = this.serializeNode(this.treeRoot);
        const htmlContent = this.generateHTMLContent(serializedRoot);
        fs.writeFileSync(filePath, htmlContent);
        console.log(`Tree visualization saved to ${filePath}`);
    }

    private generateHTMLContent(treeData: any): string {
        return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Coq Proof Tree</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            .node circle { fill: #69b3a2; }
            .node text { font: 12px sans-serif; pointer-events: none; }
            .link { fill: none; stroke: #999; stroke-opacity: 0.6; stroke-width: 2px; }
            .tooltip { position: absolute; text-align: left; padding: 5px; background: #f9f9f9; border: 1px solid #d3d3d3; border-radius: 5px; pointer-events: none; }
            .edge-label { font: 10px sans-serif; fill: #333; }
        </style>
    </head>
    <body>
        <svg width="1000" height="600"></svg>
        <div id="tooltip" class="tooltip" style="opacity:0;"></div>
        <script>
            const treeData = ${JSON.stringify(treeData, null, 2)};
            
            const svg = d3.select("svg"),
                width = +svg.attr("width"),
                height = +svg.attr("height");
    
            const g = svg.append("g").attr("transform", "translate(40,0)");
            const root = d3.hierarchy(treeData, d => d.children.map(c => c.node));
            const treeLayout = d3.tree().size([height, width - 160]);
            treeLayout(root);
    
            const links = g.selectAll(".link")
                .data(root.links())
                .enter()
                .append("path")
                .attr("class", "link")
                .attr("id", (d, i) => \`edgepath\${i}\`)
                .attr("d", d3.linkHorizontal().x(d => d.y).y(d => d.x));
    
            const edgeLabels = g.selectAll(".edge-label")
                .data(root.links())
                .enter()
                .append("text")
                .attr("class", "edge-label")
                .attr("dy", -3)
                .style("text-anchor", "middle")
                .append("textPath")
                .attr("href", (d, i) => \`#edgepath\${i}\`)
                .attr("startOffset", "50%")
                .text(d => d.target.data.stepText);
    
            const nodes = g.selectAll(".node")
                .data(root.descendants())
                .enter().append("g")
                .attr("class", "node")
                .attr("transform", d => "translate(" + d.y + "," + d.x + ")");
    
            nodes.append("circle").attr("r", 5);
    
            nodes.append("foreignObject")
                .attr("width", 200)
                .attr("height", 100)
                .append("xhtml:body")
                .style("font", "14px 'Helvetica Neue', Arial, sans-serif")
                .html(d => d.data.proofState);
    
            const tooltip = d3.select("#tooltip");
            nodes.on("mouseover", function(event, d) {
                tooltip.transition().duration(200).style("opacity", .9);
                tooltip.html(d.data.proofState)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }).on("mouseout", function() {
                tooltip.transition().duration(500).style("opacity", 0);
            });
        </script>
    </body>
    </html>`;
    }    
}
