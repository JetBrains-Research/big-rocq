# RocqStar Embeddings

Given a dataset of theorem–proof pairs \[$(s_i, p_i)$\], we train an embedder **r** that, given two statements, predicts how similar their proofs are. We fine-tune the 108M-parameter CodeBERT encoder using an InfoNCE loss with hard negatives.  
At inference time, for a new statement $(s_*)$, **r** ranks a pool of candidate premises, and we select the top-$k$ for proof generation.

* **Logging:** Uses Python’s built‐in logging (level set via log_level in config.yaml) to print training progress, losses, and warnings to the console (and any configured log files).
* **Weights & Biases:** Toggle with wandb.enabled in config.yaml; set project, entity and experiment_name to have your run automatically track hyperparameters, metrics, and system stats.
* **PyTorch Training & Checkpoints:** main.py reads config.yaml, builds DataLoader → model → optimizer/scheduler, then runs for steps iterations. Checkpoints are saved to evaluation.output_dir every save_freq steps.


## Training the model

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure 

Edit `config.yaml` to set paths, hyperparameters, and logging:

```bash
project_name: "coq-theorem-embedding"
experiment_name: "infonce-final-1"
log_level: "INFO"

dataset:
  dataset_path: "./data/"
  rankin_ds_path_statements: "./data/imm/basic/Events_statements.json"
  rankin_ds_path_references: "./sanityCheckSet/reference_premises.json"
  samples_from_single_anchor: 150
  train_split: 0.7
  val_split: 0.2
  test_split: 0.1

base_model_name: "microsoft/codebert-base"
max_seq_length: 128
embedding_dim: 768

threshold_pos: 0.3
threshold_neg: 0.65
threshold_hard_neg: 0.65
k_negatives: 4

steps: 22000
batch_size: 32
warmup_ratio: 0.1
learning_rate: 4e-6
random_seed: 52

wandb:
  enabled: false
  project: "coq-embeddings"
  entity: "YOUR_WANDB_ENTITY"
  tags: ["dyn-lr", "InfoNCE", "hard-negatives", "codebert"]

evaluation:
  output_dir: "./checkpoints/"
  save_freq: 1000
  eval_steps: 200
  evaluate_freq: 200
  k_values: [5, 10]
  f_score_beta: 1
  query_size_in_eval: 20
```

### 3. Train

```bash
python main.py --config config.yaml
```