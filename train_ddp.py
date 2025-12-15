import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import (
    init_process_group,
    destroy_process_group,
    broadcast,
    get_rank,
    barrier,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.special import softmax
from torchvision.datasets import MNIST
from circuit_builder import generate_device_aware_gate_circ
from create_circuit import create_qiskit_circ
from dataset import AugmentedZigzagDataset
from model import TQCirc, TQCeLoss

def compute_accuracy_sklearn(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy().ravel()
    return accuracy_score(y_true, preds)

def compute_auc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    y     = labels.detach().cpu().numpy().ravel()
    C = probs.shape[1]
    if C == 2:
        return roc_auc_score(y, probs[:,1])
    else:
        return roc_auc_score(y, probs, multi_class='ovr')

def ddp_setup():
    init_process_group(backend="nccl")
    # pick a seed on global rank 0, broadcast it to everyone
    if get_rank() == 0:
        seed = random.randrange(2**32)
    else:
        seed = 0
    local_rank = int(os.environ["LOCAL_RANK"])
    seed_tensor = torch.tensor([seed], device=f"cuda:{local_rank}")
    broadcast(seed_tensor, src=0)
    seed = seed_tensor.item()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#  Trainer
class Trainer:
    def __init__(self, train_loader,val_loader,test_loader, model,optimizer, scheduler,loss_fn, save_every, snapshot_path: str, early_stop_patience: int = 10,monitor: str = "val_loss"):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_loader = train_loader
        self.val_loader  = val_loader
        self.test_loader  = test_loader
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.loss_fn      = loss_fn
        self.save_every   = save_every
        self.snapshot_path = snapshot_path
        self.best_snapshot    = snapshot_path.replace(".pt", "_best.pt")
        # early-stopping bookkeeping
        self.patience    = early_stop_patience
        self.monitor     = monitor
        if   monitor == "val_loss":
             # lower is better
             self.best_metric = float("inf")
        elif monitor in ("val_acc", "val_auc"):
             # higher is better
             self.best_metric = 0.0
        else:
             raise ValueError(f"Unrecognized monitor: {monitor!r}")
        self.no_improve = 0
        self.epochs_run = 0
        if os.path.exists(snapshot_path):
            print("[Rank0] Loading snapshot")
            self._load_snapshot(self.snapshot_path)
        
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path, map_location=f"cuda:{self.local_rank}")
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"[Rank0] Resuming from epoch {self.epochs_run}")

    def _run_epoch(self, epoch):
        self.model.train()
        sampler: DistributedSampler = self.train_loader.sampler
        sampler.set_epoch(epoch)
        if self.global_rank == 0:
            bsz = next(iter(self.train_loader))[0].size(0)
            print(f"[GPU{self.global_rank}] Epoch {epoch+1} | Batchsize: {bsz} | Steps: {len(self.train_loader)}")
        for step, (x, y) in enumerate(self.train_loader):
            x = x.to(self.local_rank)
            y = y.to(self.local_rank)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()
            if step == 0 and self.global_rank == 0:
                # only print from rank 0 to avoid clutter
                acc = compute_accuracy_sklearn(logits, y)
                print(f"[Epoch {epoch+1:>2}] Step {step+1:>3}  Loss {loss:.4f}  Acc {acc:.4f}")

        self.scheduler.step()

    @torch.no_grad()
    def _validate(self, epoch):
        if self.global_rank != 0:
            return 0.0, 0.0, 0.0
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []
        for x, y in self.val_loader:
            x = x.to(self.local_rank)
            y = y.to(self.local_rank)
            logits = self.model(x)
            batch_size = x.size(0)
            total_loss += self.loss_fn(logits, y).item() * batch_size
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels).view(-1)
        avg_loss = total_loss / len(all_labels)
        avg_acc = compute_accuracy_sklearn(all_logits, all_labels)
        avg_auc   = compute_auc(all_logits, all_labels)
        print(f"→ [Val] Epoch {epoch+1:>2}  Loss {avg_loss:.4f}  Acc {avg_acc:.4f}  AUC {avg_auc:.4f}")

        # ---checkpoint on best ---
        if self.monitor == "val_loss":
            current, improved = avg_loss, (avg_loss < self.best_metric)
        elif self.monitor == "val_acc":
            current, improved = avg_acc,  (avg_acc  > self.best_metric)
        elif self.monitor == "val_auc":
            current, improved = avg_auc,  (avg_auc  > self.best_metric)
        else:
            # should never happen thanks to __init__ guard
            raise ValueError(f"Invalid monitor: {self.monitor!r}")
        
        if improved:
            self.best_metric = current
            self.no_improve = 0
            d = os.path.dirname(self.best_snapshot)
            if d:
                os.makedirs(d, exist_ok=True)

            torch.save({
                "MODEL_STATE": self.model.module.state_dict(),
                "EPOCHS_RUN": epoch+1
            }, self.best_snapshot)
            print(f"→ [Rank0] New best saved @ epoch {epoch+1}")
        else:
            self.no_improve += 1
            print(f"[Rank0] no improvement for {self.no_improve}/{self.patience} epochs")

        return avg_loss, avg_acc, avg_auc

    def _save_snapshot(self, epoch):
        # only rank 0 saves
        if self.global_rank == 0 and (epoch+1) % self.save_every == 0:
            d = os.path.dirname(self.snapshot_path)
            if d:
                os.makedirs(d, exist_ok=True)

            torch.save({
                "MODEL_STATE": self.model.module.state_dict(),
                "EPOCHS_RUN": epoch + 1,
            }, self.snapshot_path)
            print(f"[Rank0] Epoch {epoch+1} snapshot saved")

    @torch.no_grad()
    def tune_threshold(self):
        if self.global_rank != 0:
            return None

        # collect labels & per‐class probs
        all_true = []
        all_probs = []
        for x, y in self.val_loader:
            x = x.to(self.local_rank)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()  # shape [bs, C]
            all_probs.append(probs)
            all_true .append(y.cpu().numpy().ravel())

        all_probs = np.concatenate(all_probs, axis=0)  # [N, C]
        all_true  = np.concatenate(all_true,  axis=0)  # [N,]

        C = all_probs.shape[1]
        if C == 2:
            # binary: sweep threshold on class-1 prob
            best_t, best_acc = 0.5, 0.0
            for t in np.linspace(0, 1, 101):
                preds = (all_probs[:, 1] > t).astype(int)
                acc = accuracy_score(all_true, preds)
                if acc > best_acc:
                    best_acc, best_t = acc, t
            print(f"→ Tuned threshold={best_t:.2f} → Val-acc={best_acc:.4f}")
            return best_t
        else:
            # multiclass: no threshold tuning, just argmax
            print("→ Multiclass: skipping threshold tuning")
            return None

    @torch.no_grad()
    def test_with_threshold(self, threshold: float):
        """Evaluate on TEST set, using threshold if binary, or argmax if multiclass."""
        if self.global_rank != 0:
            return

        # gather test logits and labels
        all_logits = []
        all_true   = []
        total_loss = 0.0
        total_n    = 0

        for x, y in self.test_loader:
            x = x.to(self.local_rank)
            y = y.to(self.local_rank)
            logits = self.model(x)
            bs = x.size(0)
            total_loss += self.loss_fn(logits, y).item() * bs
            total_n    += bs

            all_logits.append(logits.cpu().numpy())
            all_true  .append(y.cpu().numpy().ravel())

        all_logits = np.concatenate(all_logits, axis=0)  # [N, C]
        all_true   = np.concatenate(all_true,   axis=0)  # [N,]
        avg_loss   = total_loss / total_n

        C = all_logits.shape[1]
        if C == 2 and threshold is not None:
            # binary with tuned threshold
            probs = softmax(all_logits, axis=1)[:, 1]
            preds = (probs > threshold).astype(int)
            acc   = accuracy_score(all_true, preds)
            auc   = roc_auc_score(all_true, probs)
            print(f"→ [Test @ t={threshold:.2f}] Loss={avg_loss:.4f}  Acc={acc:.4f}  AUC={auc:.4f}")
        else:
            # multiclass (or binary w/o threshold): argmax
            preds = all_logits.argmax(axis=1)
            acc   = accuracy_score(all_true, preds)
            auc   = roc_auc_score(all_true, softmax(all_logits, axis=1), multi_class='ovr')
            print(f"→ [Test] Loss={avg_loss:.4f}  Acc={acc:.4f}  AUC={auc:.4f}")

    def train(self, total_epochs):
        for epoch in range(self.epochs_run, total_epochs):
            self._run_epoch(epoch)
            barrier()
            self._validate(epoch)
            barrier()

            if self.global_rank == 0:
                stop = (self.no_improve >= self.patience)
                if stop:
                    print(f"[Rank0] Early stopping at epoch {epoch+1}")        
            else:
                stop = False
            
            stop_tensor = torch.tensor(int(stop), device = f"cuda:{self.local_rank}")
            broadcast(stop_tensor, src = 0)
            stop = bool(stop_tensor.item())
            if stop:
                barrier()
                break
            
            self._save_snapshot(epoch)
            barrier()
            self.epochs_run = epoch + 1
 
        if self.global_rank == 0:
            if os.path.exists(self.best_snapshot):
                state = torch.load(self.best_snapshot, map_location = f"cuda:{self.local_rank}") 
                self.model.module.load_state_dict(state["MODEL_STATE"])
                print(f"[Rank0] Loaded best model from {self.best_snapshot}")

            best_t = self.tune_threshold()
            print("→ [Evaluate @ best snapshot]")
            self.test_with_threshold(best_t)

            print("[Evaluate @ best snapshot @ t = 0.50]")
            self.test_with_threshold(0.5)

            #Evaluate on last checkpoint        
            if os.path.exists(self.snapshot_path):
                state = torch.load(self.snapshot_path, map_location = f"cuda:{self.local_rank}")
                self.model.module.load_state_dict(state["MODEL_STATE"])
                print(f"[Rank0] Loaded last snapshot from{self.snapshot_path}")
        
            last_t = self.tune_threshold()
            print(" --> [Evaluate @ last snapshot]")
            self.test_with_threshold(last_t)

            print("[Evaluate @ last snapshot @ t = 0.50]")
            self.test_with_threshold(0.5)
            
        barrier()

def main(save_every, num_epochs, batch_size, snapshot_path: str = "./trained_circuit_mnist10/snapshot.pt"):
    ddp_setup()
    global_rank = get_rank()
    # generate your random circuit exactly once
    
    circ_gates, gate_params, inputs_bounds, weights_bounds = generate_device_aware_gate_circ(
            num_qubits, num_embeds, num_params, param_focus, add_rotations, num_embed_cols, entangle_freq)

    print(circ_gates)
    print(gate_params)
    print(inputs_bounds)
    print(weights_bounds)

    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, random_split
    if global_rank == 0:
        mnist_full = MNIST(root = "./mnistdata_train", train = True, download = True)
        mnist_test = MNIST(root = "./mnistdata_test", train = False, download = True)
    
    barrier()

    mnist_full = MNIST(root="./mnistdata_train", train=True,  download=False)
    mnist_test = MNIST(root="./mnistdata_test",  train=False, download=False)

    all_imgs = mnist_full.data.numpy()
    all_lbls = mnist_full.targets.numpy()
    full_aug_ds = AugmentedZigzagDataset(imgs = all_imgs, lbls = all_lbls)
    print(len(full_aug_ds))
    # split: 50k train / 10k val
    train_size = len(full_aug_ds) - 10000
    val_size = 10000
    train_ds, val_ds = random_split(full_aug_ds, [train_size, val_size], generator = torch.Generator().manual_seed(42))
    print(len(train_ds))
    print(len(val_ds))
    # MNIST test (10k) → custom test ds
    
    test_imgs = mnist_test.data.numpy()
    test_lbls = mnist_test.targets.numpy()
    test_ds = AugmentedZigzagDataset(imgs = test_imgs, lbls = test_lbls)
    print(len(test_ds))
    train_data_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
              sampler=torch.utils.data.DistributedSampler(train_ds), num_workers = 4, pin_memory=True)
    val_data_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
              sampler=torch.utils.data.SequentialSampler(val_ds), num_workers = 2, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
              sampler=torch.utils.data.SequentialSampler(test_ds), num_workers = 2, pin_memory=True)
    
    if global_rank == 0:
        curr_circ_dir = os.path.dirname(snapshot_path)
        os.makedirs(curr_circ_dir, exist_ok = True)
        np.savetxt(os.path.join(curr_circ_dir, 'gates.txt'), circ_gates, fmt = "%s")
        np.save(os.path.join(curr_circ_dir, 'gate_params.npy'), np.array(gate_params, dtype = object), allow_pickle = True)   
        np.savetxt(os.path.join(curr_circ_dir, 'inputs_bounds.txt'), inputs_bounds)
        np.savetxt(os.path.join(curr_circ_dir, 'weights_bounds.txt'), weights_bounds)
        measurable_qubit = list(range(num_qubits))
        circ_creator = create_qiskit_circ(circ_gates, gate_params, inputs_bounds, weights_bounds, measurable_qubit, num_qubits)
        curr_params = np.random.uniform(0, np.pi, size = num_params)
        sample_batch, _ = next(iter(test_data_loader))
        one_sample = sample_batch[0].cpu().numpy()
        circ = circ_creator(one_sample, curr_params)
        fig_path = os.path.join(curr_circ_dir, "circuit.png")
        circ.draw(output = "mpl", filename = fig_path)

    model = TQCirc(
            circ_gates, gate_params, inputs_bounds, weights_bounds,
            num_qubits, num_classes=10, use_softmax=False, 
            quantize=False, noise_strength=0.05)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn   = TQCeLoss()
    trainer = Trainer(train_data_loader, val_data_loader, test_data_loader, model, optimizer,
           scheduler, loss_fn, save_every, snapshot_path, early_stop_patience, monitor="val_auc")
    trainer.train(num_epochs)
    destroy_process_group()

num_qubits = 16
num_embeds = 256
num_params = 512
param_focus = 2
add_rotations = True
num_embed_cols = 1
entangle_freq = 4
num_epochs = 250
save_every = 1
batch_size = 64
learning_rate = 0.005
early_stop_patience = 20

if __name__ == "__main__":
    main(save_every, num_epochs, batch_size)    




