import os
import numpy as np
import pickle as pkl
import torch
import time
from qiskit_aer import AerSimulator
import torch.nn.functional as F
from qiskit_ibm_runtime import QiskitRuntimeService, Session, EstimatorV2 as Estimator
from qiskit.compiler import transpile
from qiskit.quantum_info import SparsePauliOp
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from create_circuit import create_qiskit_circ
from model import TQCeLoss
from dataset import AugmentedZigzagDataset

def get_params_from_tq_model(model_dir: str, num_params: int) -> np.ndarray:
    ckpt = torch.load(os.path.join(model_dir, 'snapshot.pt'), map_location='cpu')
    state = ckpt['MODEL_STATE']
    model_params = np.zeros(num_params, dtype=float)

    for i in range(num_params):
        key = f"var_gates.{i}.params"
        # sanity check
        if key not in state:
            raise KeyError(f"Could not find '{key}' in MODEL_STATE keys.")
        model_params[i] = state[key].cpu().item()

    return model_params

def get_circ_params(dir_path):
    # Load inputs and weights bounds from text files
    inputs_bounds = [int(i) for i in np.genfromtxt(dir_path + '/inputs_bounds.txt')]
    weights_bounds = [int(i) for i in np.genfromtxt(dir_path + '/weights_bounds.txt')]
    
    # Read gates from text file
    gates = open(dir_path + '/gates.txt').read().split('\n')
    
    # Load gate_params from .npy file (dtype=object)
    gate_params = np.load(dir_path + '/gate_params.npy', allow_pickle=True)
    
    # Convert to list if necessary (to match the previous behavior)
    if isinstance(gate_params, np.ndarray):
        gate_params = gate_params.tolist()
    
    # Filter empty gate entries
    gates = list(filter(lambda x: x != '', gates))
    
    return gates, gate_params, inputs_bounds, weights_bounds

def run_qiskit_circ(circuit, dev, noisy_dev, num_meas_qubits, num_shots=1024,
                    mode='exp', opt_level=0):

    new_circuit = transpile(circuit, backend=noisy_dev, optimization_level=opt_level,seed_transpiler=42) #initial_layout=qubit_mapping
    observables = [SparsePauliOp("I"*(num_meas_qubits-n-1) + "Z" + "I"*n) for n in range(num_meas_qubits)]
    isa_observables = [[obs.apply_layout(new_circuit.layout)] for obs in observables]
    estimator = Estimator(mode=dev)
    estimator.options.default_shots = num_shots
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XpXm"
    estimator.options.resilience.measure_mitigation = True
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = "auto"
    estimator.options.resilience.zne_mitigation = True
    estimator.options.resilience.zne.noise_factors = (1, 3, 5)
    estimator.options.resilience.zne.extrapolator = ("exponential", "linear")
    job = estimator.run([(new_circuit, isa_observables)])
    result = job.result()
    output = result[0].data.evs.T
    return output

def tq_model_inference_on_noisy_sim_qiskit(circ_dir, device_name, num_qubits, meas_qubits, noisy_dev,
                                           x_test, y_test, save=True, num_shots=1024, transpile_opt_level=0, 
                                           results_save_dir=None):
    num_meas_qubits = len(meas_qubits)
    circ_gates, gate_params, inputs_bounds, weights_bounds = get_circ_params(circ_dir)
    print(circ_gates)
    print(gate_params)
    print(inputs_bounds)
    print(weights_bounds)
    circ_creator = create_qiskit_circ(circ_gates, gate_params, inputs_bounds,
                                      weights_bounds, meas_qubits, num_qubits)

    os.makedirs(results_save_dir, exist_ok=True)
    device_results_save_dir = os.path.join(results_save_dir, device_name)
    os.makedirs(device_results_save_dir, exist_ok=True)
    
    losses_list = []
    accs_list = []
    aucs_list = []
    senss_list= []
    specs_list= []
    f1s_list= []
    curr_run_dir = os.path.join(circ_dir)   
    print('Fetching param values from', curr_run_dir)
    curr_params = get_params_from_tq_model(curr_run_dir, weights_bounds[-1])
    val_exps = []    
    circ_list = [circ_creator(sample, curr_params) for sample in x_test]

    with open(os.path.join(device_results_save_dir, 'x_test.pkl'), 'wb') as f:
        pkl.dump(x_test, f)

    with open(os.path.join(device_results_save_dir, 'y_test.pkl'), 'wb') as f:
        pkl.dump(y_test, f)

    with Session(backend=noisy_dev) as session:
        for i in range(len(x_test)):            
            val_exps.append(
                run_qiskit_circ(
                    circ_list[i], session, noisy_dev, num_meas_qubits, num_shots,
                    'exp', transpile_opt_level,
                )
            )                         
    val_exps = np.vstack(val_exps)  
    #save the results for future check if so desired
    with open(os.path.join(device_results_save_dir, 'val_exps.pkl'), 'wb') as f:
        pkl.dump(val_exps, f)

    snap_path = os.path.join(curr_run_dir, 'snapshot.pt')
    ckpt     = torch.load(snap_path, map_location='cpu')
    state    = ckpt['MODEL_STATE']

    # extract classifier weights & bias
    W = state['classifier.weight'].cpu().numpy()   # shape [num_classes, num_qubits]
    b = state['classifier.bias'].cpu().numpy()     # shape [num_classes]

    # compute logits: [n_samples, num_classes]
    logits = val_exps.dot(W.T) + b

    logits_t = torch.from_numpy(logits)
    labels_t = torch.from_numpy(y_test).long()

    loss_val = TQCeLoss()(logits_t, labels_t).item()
    
    probs = F.softmax(logits_t, dim=1).cpu().numpy()
    C = probs.shape[1]
    if C == 2:
        preds = (probs[:,1] > best_t).astype(int)
        acc   = accuracy_score(y_test, preds)
        auc   = roc_auc_score(y_test, probs[:,1])
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)             # recall for positive class
        specificity = tn / (tn + fp)             # true negative rate
        f1 = f1_score(y_test, preds)             # F1 score for positive class
    else:
        preds = probs.argmax(axis=1)
        acc   = accuracy_score(y_test, preds)
        auc   = roc_auc_score(y_test, probs, multi_class="ovr")
        sensitivity = recall_score(y_test, preds, average="macro")
        cm = confusion_matrix(y_test, preds)
        spec_per_class = []
        for i in range(C):
            # True negatives for class i:
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            spec_per_class.append(tn / (tn + fp))
        specificity = sum(spec_per_class) / C
        # macro-averaged F1
        f1 = f1_score(y_test, preds, average="macro")

    losses_list.append(loss_val)
    accs_list.append(acc)
    aucs_list.append(auc)
    senss_list.append(sensitivity)
    specs_list.append(specificity)
    f1s_list.append(f1)
    print(f"[Inference] Loss: {loss_val:.4f} "
    f"Acc: {acc:.4f}  AUC: {auc:.4f}  "
    f"Sens: {sensitivity:.4f}  Spec: {specificity:.4f}  F1: {f1:.4f}"
    )

    if save:
        np.savetxt(os.path.join(device_results_save_dir, "val_losses.txt"), losses_list)
        np.savetxt(os.path.join(device_results_save_dir, "accs.txt"), accs_list)
        np.savetxt(os.path.join(device_results_save_dir, "aucs.txt"), aucs_list)
        np.savetxt(os.path.join(device_results_save_dir, "sensitivities.txt"), senss_list)
        np.savetxt(os.path.join(device_results_save_dir, "specificities.txt"), specs_list)
        np.savetxt(os.path.join(device_results_save_dir, "f1s.txt"), f1s_list)
        
    return losses_list, accs_list, aucs_list, senss_list, specs_list, f1s_list

def run_noisy_inference_for_tq_circuits_qiskit(circ_dir, num_qubits, meas_qubits, device_name,
                                               num_test_samples=None, num_shots=1024, transpile_opt_level=0, results_save_dir=None):

    from torchvision.datasets import MNIST, FashionMNIST
    from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, random_split
    mnist_test = MNIST(root = "./mnistdata2_test", train = False, download = True)
    test_imgs = mnist_test.data.numpy()
    test_lbls = mnist_test.targets.numpy()
    test_ds = AugmentedZigzagDataset(imgs = test_imgs, lbls = test_lbls)
    print(len(test_ds))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=len(test_ds),
              sampler=torch.utils.data.SequentialSampler(test_ds), num_workers = 2, pin_memory=True)
    x_test, y_test = next(iter(test_loader))
    x_test = x_test.numpy()
    y_test = y_test.numpy()
      
    noisy_dev = AerSimulator(device = "GPU")
    service = QiskitRuntimeService()
    
    #noisy_dev = service.backend(device_name)
  
    if results_save_dir is not None:
        if os.path.isabs(results_save_dir):
            out_dir = results_save_dir
        else:
            out_dir = os.path.join(circ_dir, results_save_dir)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = circ_dir
    
    if num_test_samples:
        sel_inds = np.random.choice(len(x_test), num_test_samples, False)
        print(sel_inds)
        x_test = x_test[sel_inds]
        y_test = y_test[sel_inds]
        
    losses_list, accs_list, aucs_list, senss_list, specs_list, f1s_list = tq_model_inference_on_noisy_sim_qiskit(
            circ_dir, device_name, num_qubits, meas_qubits,
            noisy_dev, x_test, y_test, 
            True, num_shots, transpile_opt_level, results_save_dir,
    )
           
    print(losses_list)
    print(accs_list)
    print(aucs_list)
    print(senss_list)
    print(specs_list)
    print(f1s_list)


def main():              
    meas_qubits = [i for i in range(num_meas_qubits)]
    start = time.time()      
    run_noisy_inference_for_tq_circuits_qiskit(
        circs_dir, num_qubits, meas_qubits, device_name,
        num_test_samples, num_shots, transpiler_opt_level, 
        results_save_dir)    
    
    print('Time taken', time.time() - start)

num_qubits = 16
num_meas_qubits = 16
device_name = 'ibm_kingston'
num_shots = 10000
best_t = 0.5
circs_dir = '/home/singhg12/isilon/Gurinder/Benchmark_medmnist/QCS_gpu_binary_scale/git/trained_models/mnist10/trained_circuit/'
num_test_samples = 10000
transpiler_opt_level = 3
results_save_dir = '/home/singhg12/isilon/Gurinder/Benchmark_medmnist/QCS_gpu_binary_scale/git/trained_models/mnist10/trained_circuit/inference_sim'

if __name__ == '__main__':
    main()


