import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf

from create_circuit import (
    CPhase, RXY, GPI, gpi, GPI2, MS,
    ecr, cphase, rxy, gpi2, ms,
)

class TQCirc(tq.QuantumModule):
    def __init__(self, gates, gate_params, inputs_bounds,
                 weights_bounds, num_qubits, num_classes: int = 10, use_softmax=False,
                 quantize=False, noise_strength=0.05, use_tt=False,
                 tt_input_size=None, tt_ranks=None,
                 tt_output_size=None):            
        super().__init__()
        
        trainable_mapping = {
            'ry': tq.operator.standard_gates.RY,
            'rx': tq.operator.standard_gates.RX,
            'rz': tq.operator.standard_gates.RZ,
            'cry': tq.operator.standard_gates.CRY,
            'crz': tq.operator.standard_gates.CRZ,
            'crx': tq.operator.standard_gates.CRX,
            'rxx': tq.operator.standard_gates.RXX,
            'ryy': tq.operator.standard_gates.RYY,
            'rzz': tq.operator.standard_gates.RZZ,
            'rzx': tq.operator.standard_gates.RZX,
            'cp': CPhase,
            'cphaseshift': CPhase,
            'rxy': RXY,
            'xy': RXY,
            'gpi': GPI,
            'gpi2': GPI2,
            'ms': MS
        }  

        non_trainable_mapping = {
            'ry': tqf.ry,
            'rx': tqf.rx,
            'rz': tqf.rz,
            'cry': tqf.cry,
            'crz': tqf.crz,
            'crx': tqf.crx,
            'cx': tqf.cnot,
            'cz': tqf.cz,
            'rzz': tqf.rzz,
            'rxx': tqf.rxx,
            'ryy': tqf.ryy,
            'h': tqf.hadamard,
            'rzx': tqf.rzx,
            'sx': tqf.sx,
            'x': tqf.paulix,
            'ecr': ecr,
            'v': tqf.sx,
            'cp': cphase,
            'cphaseshift': cphase,
            'rxy': rxy,
            'xy': rxy,
            'amp_enc': tq.StateEncoder(),
            'gpi': gpi,
            'gpi2': gpi2,
            'ms': ms
        }
        
        self.n_wires = num_qubits
        #self.device = tq.QuantumDevice(n_wires=self.n_wires)
        self.classifier = torch.nn.Linear(num_qubits, num_classes)
        self.use_softmax = use_softmax
        self.quantize = quantize
        
        if quantize:
            self.normalizer = torch.nn.BatchNorm1d(num_qubits, track_running_stats=False)
            #self.noise_injector = lambda x: x + noise_strength * torch.randn(x.shape)
            self.noise_injector = lambda x: x + noise_strength * torch.randn(x.shape, device=x.device)
        else:
            self.normalizer = lambda x: x
            #self.normalizer = torch.nn.BatchNorm1d(num_qubits)
            self.noise_injector = lambda x: x
        
        if use_tt:
            self.tt_layer = TTLinear(
                inp_modes=tt_input_size,
                out_modes=tt_output_size,
                tt_rank=tt_ranks
            )
        else:
            self.tt_layer = torch.nn.Identity()
        
        self.measure = tq.MeasureAll(tq.PauliZ)

        self.embed_gates = []
        self.var_gates = []
        self.ent_gates = []
        self.gate_params = []
        self.embed_flags = []
        self.param_flags = []
        self.gate_wires = []
        self.gate_list = gates
        self.inject_noise = False
        
        self.num_gates = len(gates)
        
        for i in range(len(gates)):
            self.gate_wires.append([int(j) for j in gate_params[i]])
            
            if weights_bounds[i] != weights_bounds[i + 1]:                
                self.var_gates.append(trainable_mapping[gates[i]](has_params=True, trainable=True))
                self.embed_flags.append(0)
                self.param_flags.append(1)
            else:
                self.param_flags.append(0)
                
                if inputs_bounds[i] != inputs_bounds[i + 1]:
                    self.embed_gates.append(non_trainable_mapping[gates[i]])
                    self.gate_params.append(inputs_bounds[i])
                    self.embed_flags.append(1)
                else:
                    self.ent_gates.append(non_trainable_mapping[gates[i]])
                    self.embed_flags.append(0)
                    
        self.var_gates = torch.nn.ModuleList(self.var_gates)

    def set_noise_injection(self, inject_noise):
        self.inject_noise = inject_noise
        
    def forward(self, x):
        emb_ind = 0
        ent_ind = 0
        var_ind = 0
        
        if self.inject_noise:
            x = self.noise_injector(x)
        
        #self.device.reset_states(x.shape[0])

        x = self.tt_layer(x)
  
        # Create a fresh QuantumDevice on the correct device inside forward
        q_device = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        
        for i in range(self.num_gates):
            if self.embed_flags[i]:
                if self.gate_list[i] == 'amp_enc':
                    self.embed_gates[emb_ind](q_device, x)
                    continue 
                
                self.embed_gates[emb_ind](q_device, wires=self.gate_wires[i],
                    params=x[:, self.gate_params[emb_ind]])
                emb_ind += 1
            elif self.param_flags[i]:
                self.var_gates[var_ind](q_device, wires=self.gate_wires[i])
                var_ind += 1
            else:
                self.ent_gates[ent_ind](q_device, wires=self.gate_wires[i])
                ent_ind += 1
            
        meas = self.measure(q_device)

        logits = self.classifier(meas)

        if self.use_softmax:
            meas = torch.nn.functional.log_softmax(logits, 1)

        logits = self.normalizer(logits)

        return logits

class TQCeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long().view(-1)
        return F.cross_entropy(preds, labels)
