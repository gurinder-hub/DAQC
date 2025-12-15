import numpy as np
import torch
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, CircuitInstruction
from qiskit.circuit.library import XXPlusYYGate
from torchquantum.functional import gate_wrapper
from torchquantum.operator.op_types import Operation

def gpi_matrix(params):
    phi = params.type(torch.complex64)[:, 0]

    matrix = torch.zeros(
        (2, 2), device=params.device,
        dtype=torch.complex64
    ).unsqueeze(0).repeat(params.shape[0], 1, 1)  

    matrix[:, 0, 1] = torch.exp(-1j * phi)
    matrix[:, 1, 0] = torch.exp(1j * phi)

    return matrix.squeeze(0) 

def gpi2_matrix(params):
    phi = params.type(torch.complex64)[:, 0]

    matrix = torch.eye(
        2, device=params.device,
        dtype=torch.complex64
    ).unsqueeze(0).repeat(params.shape[0], 1, 1)  

    matrix[:, 0, 1] = -1j * torch.exp(-1j * phi)
    matrix[:, 1, 0] = -1j * torch.exp(1j * phi)

    matrix *= 1 / (2 ** 0.5)

    return matrix.squeeze(0)

def ms_matrix(params):
    phi_0 = params[:, 0].type(torch.complex64)
    phi_1 = params[:, 1].type(torch.complex64)
    theta = params[:, 2].type(torch.complex64)

    cos_theta = torch.cos(theta * 0.5)
    sin_theta = torch.sin(theta * 0.5)
    phis_sum_exp = -1j * torch.exp(1j * (phi_0 + phi_1))
    phis_diff_exp = -1j * torch.exp(1j * (phi_0 - phi_1))

    matrix = torch.zeros(
        (4, 4), params.device,
        dtype=torch.complex64
    ).unsqueeze(0).repeat(params.shape[0], 1, 1)

    matrix[:, 0, 0] = cos_theta
    matrix[:, 1, 1] = cos_theta
    matrix[:, 2, 2] = cos_theta
    matrix[:, 3, 3] = cos_theta

    matrix[:, 0, 3] = phis_sum_exp * sin_theta
    matrix[:, 3, 0] = phis_sum_exp * sin_theta

    matrix[:, 1, 2] = phis_diff_exp * sin_theta
    matrix[:, 2, 1] = phis_diff_exp * sin_theta

    return matrix.squeeze(0)


def ecr_matrix(params):
    matrix = torch.tensor(
        [[0, 0, 1, 1j], [0, 0, 1j, 1], [1, -1j, 0, 0], [-1j, 1, 0, 0]],
        dtype=torch.complex64
    ) * (1 / np.sqrt(2))
    
    return matrix.squeeze(0)


def cphase_matrix(params):
    theta = params.type(torch.complex64)
    exp_theta = torch.exp(1j * theta)
    
    matrix = torch.eye(4, dtype=torch.complex64,
                         device=params.device).unsqueeze(0).repeat(exp_theta.shape[0], 1, 1)
    
    matrix[:, 3, 3] = exp_theta[:, 0]
    
    return matrix.squeeze(0)


def rxy_matrix(params):
    theta = params.type(torch.complex64)
    cos_theta = torch.cos(theta * 0.5)
    sin_theta = torch.sin(theta * 0.5)
    
    matrix = torch.eye(4, device=params.device,
                      dtype=torch.complex64).unsqueeze(0).repeat(cos_theta.shape[0], 1, 1)
    
    matrix[:, 1, 1] = cos_theta[:, 0]
    matrix[:, 2, 2] = cos_theta[:, 0]
    matrix[:, 1, 2] = -1j * sin_theta[:, 0]
    matrix[:, 2, 1] = -1j * sin_theta[:, 0]
    
    return matrix.squeeze(0)


def gpi(q_device, wires, params=None, n_wires=None, static=False,
        parent_graph=None, inverse=False, comp_method='bmm'):
    gate_wrapper(
        name='gpi', mat=gpi_matrix, method=comp_method,
        q_device=q_device, wires=wires, params=params,
        n_wires=n_wires, static=static, parent_graph=parent_graph,
        inverse=inverse
    )


def gpi2(q_device, wires, params=None, n_wires=None, static=False,
        parent_graph=None, inverse=False, comp_method='bmm'):
    gate_wrapper(
        name='gpi2', mat=gpi2_matrix, method=comp_method,
        q_device=q_device, wires=wires, params=params,
        n_wires=n_wires, static=static, parent_graph=parent_graph,
        inverse=inverse
    )


def ms(q_device, wires, params=None, n_wires=None, static=False,
        parent_graph=None, inverse=False, comp_method='bmm'):
    gate_wrapper(
        name='ms', mat=ms_matrix, method=comp_method,
        q_device=q_device, wires=wires, params=params,
        n_wires=n_wires, static=static, parent_graph=parent_graph,
        inverse=inverse
    )


def ecr(q_device, wires, params=None, n_wires=None, static=False, parent_graph=None, inverse=False, comp_method='bmm'):
    gate_wrapper(name='ecr', mat=ecr_matrix, method=comp_method,
        q_device=q_device, wires=wires, params=params,
        n_wires=n_wires, static=static,
        parent_graph=parent_graph, inverse=inverse)


def cphase(q_device, wires, params=None, n_wires=None, static=False, parent_graph=None, inverse=False, comp_method='bmm'):
    gate_wrapper(name='cp', mat=cphase_matrix, method=comp_method,
        q_device=q_device, wires=wires, params=params,
        n_wires=n_wires, static=static,
        parent_graph=parent_graph, inverse=inverse)
    
    
def rxy(q_device, wires, params=None, n_wires=None, static=False, parent_graph=None, inverse=False, comp_method='bmm'):
    gate_wrapper(name='rxy', mat=rxy_matrix, method=comp_method,
        q_device=q_device, wires=wires, params=params,
        n_wires=n_wires, static=static,
        parent_graph=parent_graph, inverse=inverse)
    

class GPI(Operation):
    num_params = 1
    num_wires = 1
    func = staticmethod(gpi)
    
    @classmethod
    def _matrix(cls, params):
        return gpi_matrix(params)
    

class GPI2(Operation):
    num_params = 1
    num_wires = 1
    func = staticmethod(gpi2)
    
    @classmethod
    def _matrix(cls, params):
        return gpi2_matrix(params)
    

class MS(Operation):
    num_params = 3
    num_wires = 2
    func = staticmethod(ms)
    
    @classmethod
    def _matrix(cls, params):
        return ms_matrix(params)


class RXY(Operation):
    num_params = 1
    num_wires = 2
    func = staticmethod(rxy)
    
    @classmethod
    def _matrix(cls, params):
        return rxy_matrix(params)
    
    
class CPhase(Operation):
    num_params = 1
    num_wires = 2
    func = staticmethod(cphase)
    
    @classmethod
    def _matrix(cls, params):
        return cphase_matrix(params)

    
def add_amp_encoding(circ, data):
    num_qubits = circ.num_qubits
    
    state = np.zeros(2 ** num_qubits)
    state[:len(data)] = data
    
    state /= np.sqrt(np.sum(np.power(data, 2)))
    
    circ.initialize(state, [i for i in range(num_qubits)])

def create_qiskit_circ(gates, gate_params, inputs_bounds, weights_bounds, measured_qubits, num_qubits, unbound=False):
    mapping = {
        'ry': lambda circ, data, qubit_inds: circ.ry(data[0], qubit_inds[0]),
        'rx': lambda circ, data, qubit_inds: circ.rx(data[0], qubit_inds[0]),
        'rz': lambda circ, data, qubit_inds: circ.rz(data[0], qubit_inds[0]),
        'cx': lambda circ, data, qubit_inds: circ.cx(qubit_inds[0], qubit_inds[1]),
        'cz': lambda circ, data, qubit_inds: circ.cz(qubit_inds[0], qubit_inds[1]),
        'cry': lambda circ, data, qubit_inds: circ.cry(data[0], qubit_inds[0], qubit_inds[1]),
        'crz': lambda circ, data, qubit_inds: circ.crz(data[0], qubit_inds[0], qubit_inds[1]),
        'crx': lambda circ, data, qubit_inds: circ.crx(data[0], qubit_inds[0], qubit_inds[1]),
        'ryy': lambda circ, data, qubit_inds: circ.ryy(data[0], qubit_inds[0], qubit_inds[1]),
        'rzz': lambda circ, data, qubit_inds: circ.rzz(data[0], qubit_inds[0], qubit_inds[1]),
        'zz': lambda circ, data, qubit_inds: circ.rzz(data[0], qubit_inds[0], qubit_inds[1]),
        'rxx': lambda circ, data, qubit_inds: circ.rxx(data[0], qubit_inds[0], qubit_inds[1]),
        'h': lambda circ, data, qubit_inds: circ.h(qubit_inds[0]),
        's': lambda circ, data, qubit_inds: circ.s(qubit_inds[0]),
        'x': lambda circ, data, qubit_inds: circ.x(qubit_inds[0]),
        'y': lambda circ, data, qubit_inds: circ.y(qubit_inds[0]),
        'z': lambda circ, data, qubit_inds: circ.z(qubit_inds[0]),
        'sx': lambda circ, data, qubit_inds: circ.sx(qubit_inds[0]),
        'rxy': lambda circ, data, qubit_inds: circ.append(CircuitInstruction(XXPlusYYGate(data[0]), qubit_inds, [])),
        'xy': lambda circ, data, qubit_inds: circ.append(CircuitInstruction(XXPlusYYGate(data[0]), qubit_inds, [])),
        'cp': lambda circ, data, qubit_inds: circ.cp(data[0], qubit_inds[0], qubit_inds[1]),
        'rzx': lambda circ, data, qubit_inds: circ.rzx(data[0], qubit_inds[0], qubit_inds[1]),
        'amp_enc': lambda circ, data, qubit_inds: add_amp_encoding(circ, data),
        'cphaseshift': lambda circ, data, qubit_inds: circ.cp(data[0], qubit_inds[0], qubit_inds[1]),
        'ecr': lambda circ, data, qubit_inds: circ.ecr(qubit_inds[0], qubit_inds[1]),
        'v': lambda circ, data, qubit_inds: circ.sx(qubit_inds[0])
    }
    
    circuit = QuantumCircuit(num_qubits, len(measured_qubits))
    input_params = [Parameter('x_{}'.format(i)) for i in range(inputs_bounds[-1])]
    var_params = [Parameter('t_{}'.format(i)) for i in range(weights_bounds[-1])]
    
    for i, gate in enumerate(gates):
        data_in = []
        data_in.append(input_params[inputs_bounds[i]: inputs_bounds[i + 1]])
        data_in.append(var_params[weights_bounds[i]: weights_bounds[i + 1]])

        data_in = data_in[0] + data_in[1]
        mapping[gate](circuit, data_in, gate_params[i])     
        
    for i in range(len(measured_qubits)):
        circuit.measure(measured_qubits[i], i)
    
    def qiskit_qnn(inputs, weights): 
        param_mapping = dict()
    
        for i in range(len(inputs)):
            param_mapping[input_params[i]] = inputs[i]
            
        for i in range(len(weights)):
            param_mapping[var_params[i]] = weights[i]
        
        return circuit.assign_parameters(param_mapping)
    
    if unbound:
        return circuit
    else:
        return qiskit_qnn

