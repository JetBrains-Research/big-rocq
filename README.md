# RocqStar Proof Retrieval
All submodules contain their own README files with more detailed instructions.

## Components

- **[big-rocq](./big-rocq/README.md)**  
  Proof‐augmentation utility that expands a Rocq project by transforming and recombining proof subtrees into new theorems.

- **[proof-embeddings](./proof-embeddings/README.md)**  
  Code for learning statement embeddings.

- **[ranker-server](./ranker-server/README.md)**  
  Python server wrapping the trained embedder for real‐time statement similarity queries.

- **[CoqPilot+RocqStar](./CoqPilot+RocqStar/)**  
  Patch and integration for the CoqPilot VSCode plugin to include RocqStar ranking support.

- **[embedder-training-dataset](./proof-embeddings/data/)**  
  Dataset used for training the proof embedder. It contains a collection of theorem–proof pairs, that we mined from the Coq projects. 