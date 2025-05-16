# RocqStar Proof Retrieval

A monorepo containing all code we have developed for the RocqStar proof retriever part of the paper **RocqStar: Leveraging Similarity-driven Retrieval and Agentic Systems for Rocq generation**. All subfolders contain their own README files with more detailed instructions.

## Components

- **[big-rocq](./big-rocq/README.md)**  
  Proof‐augmentation utility that expands a Rocq project by transforming and recombining proof subtrees into new theorems.

- **[proof-embeddings](./proof-embeddings/README.md)**  
  Code for learning statement embeddings.

- **[ranker-server](./ranker-server/README.md)**  
  Python server wrapping the trained embedder for real‐time statement similarity queries.

- **[CoqPilot+RocqStar](./CoqPilot+RocqStar/)**  
  Patch and integration for the CoqPilot VSCode plugin to include RocqStar ranking support.

- **[experiments](./experiments/)**  
  Code to reproduce initial experiments from Section 2 of the paper. 