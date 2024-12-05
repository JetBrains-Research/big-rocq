import * as fs from "fs";

import { CoqProofTreeNode } from "./coqProofTreeNode";
import { ppGoal } from "./proofStatePrinters";
import { Theorem } from "../coqParser/parsedTypes";
import { constructTheoremWithProof, theoremDatasetSampleToString } from "./proofBuilder";

export class TreeVisualizer {
    constructor(
        private treeRoot: CoqProofTreeNode,
        private theoremName: string,
        private contextTheorems: Theorem[]
    ) {}

    getSubtreeProof(
        treeNode: CoqProofTreeNode,
    ): string | undefined {
        if (treeNode.proofState) {
            const sample = constructTheoremWithProof(
                treeNode.proofState,
                treeNode.collectSubtreeEdges(),
                this.theoremName,
                this.contextTheorems
            );
            return theoremDatasetSampleToString(sample);
        } else {
            return undefined;
        }
    }

    private serializeNode(
        treeNode: CoqProofTreeNode,
        edgeLabel: string = ""
    ): any {
        return {
            name: treeNode.proofState
                ? ppGoal(treeNode.proofState, "html")
                : "Empty",
            subtreeProof: this.getSubtreeProof(treeNode),
            edgeLabel: edgeLabel,
            children: treeNode.children.map(([childNode, edge]) =>
                this.serializeNode(childNode, edge.text)
            ),
        };
    }

    public drawToFile(filePath: string) {
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
        <script src="https://cdn.jsdelivr.net/gh/google/code-prettify@master/loader/run_prettify.js"></script>
        <style>
            .node foreignObject { overflow: visible; }
            .link { fill: none; stroke: black; }
            .edgelabel { 
              font: 14px sans-serif; 
              fill: black;
              font-weight: light; }
            .tooltip {
                position: absolute;
                text-align: left;
                padding: 5px;
                background: #f9f9f9;
                border: 1px solid #d3d3d3;
                border-radius: 5px;
                pointer-events: none;
                font: 14px sans-serif;
                max-width: 300px;
                word-wrap: break-word;
            }
            .popup {
                position: absolute;
                background: white;
                border: 2px solid #333;
                border-radius: 5px;
                padding: 12px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                z-index: 10;
                white-space: pre-wrap; 
            }
            .popup h4 {
                margin: 0;
                padding: 2px 0;
                padding-bottom: -30px;
                padding-top: -73px;
                margin-bottom: -30px;
                margin-top: -73px;
                font-size: 16px;
            }
            .popup pre {
                font-size: 14px;
                margin-bottom: -55px;
                padding-bottom: -55px;
            }
            .popup-close {
                float: right;
                cursor: pointer;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <svg width="1000" height="600"></svg>
        <div id="tooltip" class="tooltip" style="opacity:0;"></div>
        <div id="popup" class="popup" style="display:none;">
            <span class="popup-close" onclick="document.getElementById('popup').style.display='none';">×</span>
            <div id="popup-content"></div>
        </div>
        <script>
            const data = ${JSON.stringify(treeData, null, 2)};
    
            const width = 1000;
            const height = 600;
    
            const svg = d3.select("svg")
                .attr("width", width)
                .attr("height", height)
                .append("g")
                .attr("transform", "translate(50, 50)");
    
            const root = d3.hierarchy(data);
    
            const treeLayout = d3.tree().size([height - 100, width - 100]);
            treeLayout(root);
    
            const link = svg.selectAll(".link")
                .data(root.links())
                .enter()
                .append("g")
                .attr("class", "link");
    
            link.append("path")
                .attr("d", d3.linkHorizontal()
                    .x(d => d.y)
                    .y(d => d.x))
                .attr("stroke", "#999")
                .attr("stroke-width", 2)
                .attr("fill", "none");
    
            link.append("text")
                .attr("dy", -5)
                .attr("class", "edgelabel")
                .attr("x", d => (d.source.y + d.target.y) / 2)
                .attr("y", d => (d.source.x + d.target.x) / 2)
                .style("text-anchor", "middle")
                .text(d => d.target.data.edgeLabel);
    
            const node = svg.selectAll(".node")
                .data(root.descendants())
                .enter()
                .append("g")
                .attr("class", "node")
                .attr("transform", d => \`translate(\${d.y},\${d.x})\`);
    
            node.append("circle")
                .attr("r", 10)
                .attr("fill", d => d.data.name === "Empty" ? "#ff6961" : "#69b3a2");
    
            const tooltip = d3.select("#tooltip");
            const popup = document.getElementById("popup");
            const popupContent = document.getElementById("popup-content");
    
            node.on("mouseover", function(event, d) {
                tooltip.transition().duration(200).style("opacity", .9);
                tooltip.html(d.data.name)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }).on("mouseout", function() {
                tooltip.transition().duration(500).style("opacity", 0);
            }).on("click", function(event, d) {
                popup.style.display = "block";
                popupContent.innerHTML = \`
                    <h4>Subtree Proof</h4>
                    <pre class="prettyprint">\${d.data.subtreeProof || "No proof available."}</pre>
                \`;
                popup.style.left = (event.pageX + 10) + "px";
                popup.style.top = (event.pageY + 10) + "px";
            });
        </script>
    </body>
    </html>`;
    }    
}
