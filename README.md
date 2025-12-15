# Domain-Aware Quantum Circuit for QML

This repository implements a **domain- and device-aware quantum circuit** for quantum machine learning (QML) for image classification task.

The pipeline combines:

- **TorchQuantum** for quantum simulation and training.
- **Qiskit** for circuit export, visualization, and (optionally) execution via Aer / IBM Runtime Estimator.
- **PyTorch Distributed Data Parallel (DDP)** for multi-GPU training.
- A **zigzag-based classical feature extractor** that maps 28×28 images into a 1D feature vector used as quantum circuit inputs.

## What “domain-aware” and “device-aware” mean

- **Domain-aware**: MNIST images are downsampled to 16×16, split into 4×4 blocks, and scanned in **zigzag order**. This preserves local image structure and maps it into a 1D angle vector used to drive rotation gates in the circuit.
- **Device-aware**: The circuit is built from a gate set tailored to superconducting hardware:
  - Native 1-qubit gates: `rz` (optionally `rx`, `ry` when `add_rotations=True`).
  - Native 2-qubit gates: `ecr` with a ring entanglement pattern matching typical nearest-neighbor connectivity.
  This layout is exported to Qiskit using the same gate set used in training.

---

## Repository Structure

```text
.
├── train_ddp.py          # DDP training loop, metrics, checkpoints
├── circuit_builder.py    # Gate set + stochastic, device-aware circuit sampler
├── create_circuit.py     # Qiskit + TorchQuantum custom gates and circuit exporter
├── dataset.py            # MNIST → zigzag feature dataset (domain-aware embedding)
├── model.py              # TorchQuantum circuit model (TQCirc) + cross-entropy loss
├── inference.py          # Inference of trained circuits via Qiskit (sim / hardware)
├── sbatch.txt            # Example Slurm script wrapping torchrun for multi-GPU training
└── README.md
```

---

## Installation

Tested with:

- Python 3.11.11  
- PyTorch with CUDA and distributed support  
- TorchQuantum 0.1.8  
- Qiskit 1.2.4 (plus `qiskit-aer`, `qiskit-ibm-runtime` for simulator / hardware runs)  
- torchvision, scikit-learn, scipy, numpy, matplotlib  

---

## Training

The main training script is `train_ddp.py`. It:

- Builds a **domain and device-aware circuit** using `circuit_builder.py`.
- Instantiates the `TQCirc` model from `model.py`.
- Trains with DDP and saves checkpoints under `./trained_circuit/` (by default).

Single node, 4 GPUs:

```bash
torchrun --nproc_per_node=4 train_ddp.py
```

Default behavior:

- Uses `DistributedSampler` for the training set.
- Tracks validation metrics (loss, accuracy, AUC) on rank 0.
- Applies early stopping.
- Saves:
  - `trained_circuit/snapshot.pt`
  - `trained_circuit/snapshot_best.pt`
  - Circuit metadata: `gates.txt`, `gate_params.npy`, `inputs_bounds.txt`, `weights_bounds.txt`
  - A circuit diagram `circuit.png` (from the Qiskit circuit exporter).

On a Slurm cluster, you can adapt and submit:

```bash
sbatch sbatch.txt
```

after editing resource requests and paths in `sbatch.txt`.

---

## Inference (Qiskit simulator / hardware)

`inference.py` runs **post-training inference** with the same circuit via Qiskit:

- Rebuilds the Qiskit circuit from:
  - `gates.txt`
  - `gate_params.npy`
  - `inputs_bounds.txt`
  - `weights_bounds.txt`
- Loads the trained parameters from `snapshot.pt`.
- Evaluates the model using an estimator (AerSimulator or IBM backend) and reports:
  - Loss, Accuracy, AUC, Sensitivity, Specificity, F1.

To use it:

1. Ensure you have a trained circuit directory, e.g.:

   ```text
   trained_circuit/
     ├── snapshot.pt
     ├── gates.txt
     ├── gate_params.npy
     ├── inputs_bounds.txt
     └── weights_bounds.txt
   ```

2. Edit the configuration at the bottom of `inference.py`:

   - `circs_dir` – path to the trained circuit directory.  
   - `results_save_dir` – directory where metrics and raw expectations will be saved.  
   - `device_name` – Qiskit backend name (e.g. `ibm_kingston` or a simulator).  
   - `num_test_samples` – number of test samples to run.  
   - `num_shots`, `transpiler_opt_level` – Qiskit execution parameters.  
   - `best_t` – decision threshold for binary classification (if applicable).

3. Run:

```bash
python inference.py
```

---

## Citation

If you use this code, please cite the associated manuscript on **domain-aware quantum circuit for QML**.
