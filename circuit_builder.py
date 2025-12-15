import math
import random
import numpy as np

def gate_type() -> tuple[list[list[str]], list[list[int]]]:
    # your chosen 1q set
    one_q = ["rz"]
    # your chosen 2q set
    two_q = ["ecr"]
    one_params = [1]
    two_params = [0]
    basis_gates = [one_q, two_q]
    gate_param_nums = [one_params, two_params]
    return basis_gates, gate_param_nums

def generate_softmax_dist(values: np.ndarray, temp: float) -> np.ndarray:
    logits = values / temp
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    return exps / np.sum(exps)

def generate_device_aware_gate_circ(
    num_qubits: int,
    num_embed_gates: int,
    num_var_params: int             = 0,
    param_focus: float              = 2.0,
    add_rotation_gates: bool        = False,
    num_embed_cols: int    = 1,
    entangle_freq: int     = 4,
) -> tuple[list[str], list[list[int]], list[int], list[int]]:

    basis_gates, gate_param_nums = gate_type()
    one_q, two_q       = basis_gates
    one_params, two_params = gate_param_nums

    # add rotations
    if add_rotation_gates:
        for r in ('rx','ry','rz'):
            if r not in one_q:
                one_q.append(r)
                one_params.append(1)

    # sampling weights
    one_w = generate_softmax_dist(param_focus*np.array(one_params), 1.0)
    two_w = generate_softmax_dist(param_focus*np.array(two_params), 1.0)
    param_gates   = [g for g,p in zip(one_q, one_params) if p>0]
    param_weights = [w for g,w,p in zip(one_q, one_w, one_params) if p>0]

    circ_gates     = []
    gate_params    = []
    inputs_bounds  = [0]
    weights_bounds = [0]

    # determine number of cycles from embedding
    cycles = math.ceil(num_embed_gates / (num_embed_cols * num_qubits))

    # compute trainable counts
    full_train_cols = num_var_params // num_qubits
    leftover_train = num_var_params % num_qubits

    base_full = full_train_cols // cycles
    extra_cycles = full_train_cols % cycles

    embed_remaining = num_embed_gates
    ent_pairs = [(i, (i + 1) % num_qubits) for i in range(num_qubits)]

    # run interleaved cycles
    for cycle in range(cycles):
        # EMBED
        for _ in range(num_embed_cols):
            if embed_remaining <= 0:
                break
            to_place = min(num_qubits, embed_remaining)
            for q in range(to_place):
                g = random.choices(param_gates, weights=param_weights)[0]
                circ_gates.append(g)
                gate_params.append([q])
                inputs_bounds.append(inputs_bounds[-1] + 1)
                weights_bounds.append(weights_bounds[-1])
            embed_remaining -= to_place

        # ENTANGLE if entangle_freq is set and rep % entangle_freq == 0
        if entangle_freq is not None and (cycle % entangle_freq) == 0:
            for (c,t) in ent_pairs:
                # pick one of the native 2-Q gates
                g2 = random.choices(two_q, weights=two_w)[0]
                circ_gates.append(g2)
                gate_params.append([c,t])
                inputs_bounds.append(inputs_bounds[-1])
                weights_bounds.append(weights_bounds[-1])

        # TRAINABLE
        # full-base columns
        for _ in range(base_full):
            for q in range(num_qubits):
                g = random.choices(param_gates, weights=param_weights)[0]
                circ_gates.append(g)
                gate_params.append([q])
                inputs_bounds.append(inputs_bounds[-1])
                weights_bounds.append(weights_bounds[-1] + 1)
        # extra column on early cycles
        if cycle < extra_cycles:
            for q in range(num_qubits):
                g = random.choices(param_gates, weights=param_weights)[0]
                circ_gates.append(g)
                gate_params.append([q])
                inputs_bounds.append(inputs_bounds[-1])
                weights_bounds.append(weights_bounds[-1] + 1)
        # leftover on exactly extra_cycles
        if cycle == extra_cycles and leftover_train > 0:
            for i in range(leftover_train):
                g = random.choices(param_gates, weights=param_weights)[0]
                circ_gates.append(g)
                gate_params.append([i])
                inputs_bounds.append(inputs_bounds[-1])
                weights_bounds.append(weights_bounds[-1] + 1)  

    return circ_gates, gate_params, inputs_bounds, weights_bounds
