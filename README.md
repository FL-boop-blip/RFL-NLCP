RFL-NLCP: Robust Federated Learning with Non-IID Data and Limited Client Participation

Overview
- This repository implements RFL-NLCP (Robust Federated Learning with non-IID data and Limited Client Participation) to improve convergence and robustness under highly non-IID data and partial client participation.
- Includes multiple baselines (FedAvg, FedDyn, SCAFFOLD, FedSpeed, FedSMOO, FedTOGA, FedVRA) and our method RFLNLCP. Supports image and text datasets with IID/Dirichlet/Pathological partitions.
- Core ideas (RFLNLCP):
  - In round 0, each client runs a lightweight local training to collect local updates, which are reduced by PCA and clustered via KMeans.
  - In subsequent rounds, client sampling is adjusted with per-cluster enlarge factors to upweight underrepresented clusters, improving global generalization.
  - Clients use ESAM for robust local optimization; the server maintains dual variables h and performs a dual-corrected global update (akin to FedDyn/FedSMOO).



Environment and Dependencies
- Python ≥ 3.8; CUDA-enabled PyTorch recommended.
- See `requirements.txt` for a consolidated list. Key packages: `torch`, `torchvision`, `numpy`, `scikit-learn`, `scipy`, `tqdm`, `matplotlib`, `seaborn`, `Pillow`, `thop`.

Data Preparation
- MNIST/CIFAR are auto-downloaded to `./Data/Raw`.

Quick Start
- Example 1 (RFLNLCP, CIFAR-10, Dirichlet non-IID, α=0.6):
  - `python train.py --method RFLNLCP --dataset CIFAR10 --non-iid --split-rule Dirichlet --split-coef 0.6 --model ResNet18 --total-client 100 --active-ratio 0.1 --comm-rounds 1000 --local-epochs 5 --batchsize 50 --local-learning-rate 0.1 --lamb 0.1`
- Example 2 (FedAvg, IID):
  - `python train.py --method FedAvg --dataset CIFAR10 --model ResNet18 --comm-rounds 200 --local-epochs 5 --batchsize 50`
- Or use the script: `bash run.sh` (defaults to RFLNLCP).

Parameters (selected)
- Data: `--dataset {mnist,CIFAR10,CIFAR100}`, `--data-file ./`, `--out-file out/`
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
- Default loss is cross-entropy; `_validate_` adds an L2 term with `weight_decay` for stability.
- GPU is recommended; CPU fallback is supported.
