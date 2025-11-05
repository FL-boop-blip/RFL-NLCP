RFL-NLCP: Robust Federated Learning with Non-IID Data and Limited Client Participation

Overview
- This repository implements RFL-NLCP (Robust Federated Learning with non-IID data and Limited Client Participation) to improve convergence and robustness under highly non-IID data and partial client participation.
- Includes multiple baselines (FedAvg, FedDyn, SCAFFOLD, FedSpeed, FedSMOO, FedTOGA, FedVRA) and our method RFLNLCP. Supports image and text datasets with IID/Dirichlet/Pathological partitions.
- Core ideas (RFLNLCP):
  - In round 0, each client runs a lightweight local training to collect local updates, which are reduced by PCA and clustered via KMeans.
  - In subsequent rounds, client sampling is adjusted with per-cluster enlarge factors to upweight underrepresented clusters, improving global generalization.
  - Clients use ESAM for robust local optimization; the server maintains dual variables h and performs a dual-corrected global update (akin to FedDyn/FedSMOO).

Project Structure
- `train.py`: Entry point. Parses CLI args, builds dataset partitions and model, selects server algorithm, and runs training/evaluation.
- `server/`: Server-side implementations.
  - `server/server.py`: Base `Server` class with the main FL loop, client activation, evaluation, logging and saving.
  - `server/RFLNLCP.py`: RFLNLCP server with client clustering, cluster-aware sampling, and dual-corrected aggregation.
  - Other algorithms: `FedAvg.py`, `FedDyn.py`, `SCAFFOLD.py`, `FedSpeed.py`, `FedSMOO.py`, `FedTOGA.py`, `FedVRA.py`.
- `client/`: Client-side implementations.
  - `client/client.py`: Base `Client` with local training and returning local updates/model params.
  - Algorithm-specific clients: `fedavg.py`, `feddyn.py`, `fedsmoo.py`, `rflnlcp.py`.
- `dataset.py`: Dataset preparation and partitioning (supports `mnist`, `CIFAR10`, `CIFAR100`, `AG_News`; partitions: `iid`, `Dirichlet`, `Pathological`).
- `models.py`: Model zoo (e.g., `LeNet`, `ResNet18` with GroupNorm, `ResNet18_100`, `AG_News_NN`, fusion/sparse variants).
- `utils.py`: Utilities (parameter vectorization/loading, distillation KL, distribution metrics, pruning target collection, etc.).
- `utils_models.py`: Custom layers/ops (SWAT/SparsyFed series, spectral norm handler, SE/Res blocks, etc.).
- `optimizer/`: Optimizers and SAM variants (`ESAM.py`, `DRegSAM.py`, `SAM.py`, etc.), with `fused_adan/`.
- `run.sh`: Example script (defaults to `RFLNLCP` on CIFAR-10 non-IID).

Key Modules and Functions
1) Entry: `train.py`
- CLI arguments (selected):
  - Dataset: `--dataset {mnist, CIFAR10, CIFAR100, AG_News}`
  - Model: `--model {mnist_2NN, ResNet18, ResNet18P, ResNet18_100, AG_News_NN, LeNet, ResNet18_sparsy, ResNet18_fusion, LeNet_fusion}`
  - Partition: `--non-iid`, `--split-rule {Dirichlet, Pathological, iid}`, `--split-coef` (Dirichlet α or Pathological class count c)
  - Participation: `--active-ratio`, `--total-client`
  - Training: `--comm-rounds`, `--local-epochs`, `--batchsize`, `--weight-decay`, `--local-learning-rate`, `--global-learning-rate`, `--lr-decay`
  - Others: `--seed`, `--cuda`, `--data-file`, `--out-file`, `--save-model`, `--use-RI`
  - Method: `--method {FedAvg, FedDyn, SCAFFOLD, FedSpeed, FedSMOO, FedTOGA, FedVRA, RFLNLCP}`
  - RFLNLCP: `--num_cluster` (clusters), `--beta3` (enlarge scale), `--lamb` (dual regularization), `--rho` (SAM radius)
- Flow:
  - Build `DatasetObject` (IID or non-IID partition)
  - `model_func = lambda: client_model(args.model, classes)`
  - Select server (e.g., `RFLNLCP`), instantiate, and call `server.train()`

2) Base server: `server/server.py` → `Server`
- `__init__`: Initialize global model, client parameter matrix, update matrix, logs, and output directories
- `_activate_clients_`: Randomly sample active clients by `active_ratio`
- `_validate_`/`_test_`: Evaluate loss/accuracy on test set and log divergence (E||wi − w||)
- `train`: Standard FL loop (sample → broadcast → local train → aggregate → test → lr decay → timing → save)
- Hooks for subclasses: `process_for_communication`, `global_update`, `postprocess`

3) RFLNLCP server: `server/RFLNLCP.py`
- State:
  - `h_params_list`: per-client dual variable vectors
  - `comm_vecs['Local_dual_correction']`: to clients, based on `h - w`
- Key methods:
  - `_cluster_clients_`: run one local training per client, collect `local_update_list`, reduce by PCA, cluster by KMeans, return labels and counts
  - `_select_clients_`: sample clients, compute per-cluster `enlarge` factors to upweight minority clusters
  - `process_for_communication`: send init params (optional RI: `w + beta*(w - w_i_prev)`) and dual correction
  - `global_update`: `Averaged_model + mean(h)`
  - `postprocess`: scale dual update by `d` (enlarge factor): `h_i += d * Δw_i`

4) Clients: `client/client.py`, `client/rflnlcp.py`
- Base `Client.train`: cross-entropy training, returns
  - `local_update_list = last_params - Params_list`
  - `local_model_param_list = last_params`
- RFLNLCP client (`rflnlcp.py`):
  - Optimizer: `ESAM` with SGD base optimizer
  - Extra regularization: quadratic penalty on `w + (h - w)` with `0.5*lamb*||·||^2`
  - Text support: `AG_News` with `lengths`

5) Datasets and partitioning: `dataset.py`
- `DatasetObject.set_data`:
  - Download/load standard datasets (MNIST/CIFAR)
  - Partitions:
    - `iid`: equal and uniform
    - `Dirichlet`: sample α-priors per client
    - `Pathological`: c classes per client
  - Outputs: `client_x/client_y` (+ `client_l` for text), `test_x/test_y`
  - Visualization: `visualize_cifar10_distributions(...)`
- Also includes synthetic data generation and dataset wrappers

6) Models: `models.py`
- `client_model(name, n_cls, ...)` builds:
  - Vision: `LeNet`, `ResNet18`, `ResNet18_100`, fusion/sparse variants (GroupNorm replaces some BN)
  - Text: `AG_News_NN` (average embedding + linear classifier)
- `count_parameters(model, dtype)`: counts trainable params and memory (MB)

7) Utilities: `utils.py`, `utils_models.py`
- Parameters: `get_mdl_params`, `set_client_from_params`, `param_to_vector`, `get_params_list_with_shape`
- Distillation: `DistillKL(T)`; distribution metrics: `get_distribution_difference(...)`
- Sparse/custom layers: `SWAT*`, `SparsyFed*`, `SpectralNormHandler`, `SEBlock`, `ResidualBlock`, etc.

Environment and Dependencies
- Python ≥ 3.8; CUDA-enabled PyTorch recommended.
- Install dependencies:
  - `pip install -r requirements.txt`

Data Preparation
- MNIST/CIFAR are auto-downloaded to `./Data/Raw`.
- AG_News: place the official CSV under `./Data/Raw/ag_news_csv` and ensure an `AG_News` loader is available (code imports `from AG_News import *`).

Quick Start
- Example 1 (RFLNLCP, CIFAR-10, Dirichlet non-IID, α=0.6):
  - `python train.py --method RFLNLCP --dataset CIFAR10 --non-iid --split-rule Dirichlet --split-coef 0.6 --model ResNet18 --total-client 100 --active-ratio 0.1 --comm-rounds 1000 --local-epochs 5 --batchsize 50 --local-learning-rate 0.1 --lamb 0.1`
- Example 2 (FedAvg, IID):
  - `python train.py --method FedAvg --dataset CIFAR10 --model ResNet18 --comm-rounds 200 --local-epochs 5 --batchsize 50`
- Or use the script: `bash run.sh` (defaults to RFLNLCP).

Parameters (selected)
- Data: `--dataset {mnist,CIFAR10,CIFAR100,AG_News}`, `--data-file ./`, `--out-file out/`
- Partition: `--non-iid`, `--split-rule {Dirichlet, Pathological}`, `--split-coef 0.6`
- Clients: `--total-client 100`, `--active-ratio 0.1`
- Training: `--comm-rounds 1000`, `--local-epochs 5`, `--batchsize 50`
- Learning rate: `--local-learning-rate 0.1`, `--global-learning-rate 1.0`, `--lr-decay 0.998`
- Optimization/regularization: `--weight-decay 0.001`, `--lamb 0.1`, `--rho 0.1~2.0`
- Others: `--cuda 0`, `--seed 20`, `--use-RI`

Outputs and Logs
- Output root: `{out}/{method}/T={comm_rounds}/{dataset}-{split}-{arg}-{n_client}/active-{ratio}`
- Saved artifacts:
  - `Performance/tst-{method}.npy`: per-round test `[loss, acc]`
  - `Divergence/divergence-{method}.npy`: per-round divergence
  - `Time/time-{method}.npy`: per-round time
  - `Model/best.pth`: best test-accuracy server weights
  - `summary.txt`: rounds, avg time, best accuracy and round

Notes
- For `AG_News`, ensure the loader (`AG_News.py`) and dataset directory exist.
- Default loss is cross-entropy; `_validate_` adds an L2 term with `weight_decay` for stability.
- GPU is recommended; CPU fallback is supported.
