import itertools as it
from matplotlib import gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml
import qutip
import scipy as sp
from tqdm.auto import tqdm

rng = np.random.default_rng()

def sample_coupling_matrix(rows, cols):
    qubits = rows * cols
    
    # Create a 2D Lattice
    edges = [
        (si, sj) for (si, sj) in it.combinations(range(qubits), 2)
        if ((sj % cols > 0) and sj - si == 1) or sj - si == cols
    ]
    
    # sample edge weights uniformly at random from [0, 2]
    edge_weights = rng.uniform(0, 2, size=len(edges))
    
    coupling_matrix = np.zeros((qubits, qubits))
    for (i, j), w in zip(edges, edge_weights):
        coupling_matrix[i, j] = coupling_matrix[j, i] = w
        
    return coupling_matrix

# define the system size and lattice geometry
rows, cols = 4, 4
wires = rows * cols

# sample a coupling matrix
J = sample_coupling_matrix(rows, cols)


# create graph object
graph = nx.from_numpy_array(np.matrix(J), create_using=nx.DiGraph)
graph = nx.relabel_nodes(graph, {i: i + 1 for i in graph.nodes})
pos = {i: ((i-1) % cols, -((i-1) // cols)) for i in graph.nodes()}

# make edge widths proportional to edge weight
edge_widths = [
    (x + 1.5) ** 2 for x in list(nx.get_edge_attributes(graph, "weight").values())
]

# extract edge weights for colouring
edges, weights = zip(*nx.get_edge_attributes(graph,'weight').items())

plt.figure(figsize=(cols / 1.5, rows / 1.5))
nx.draw(
    graph, pos, node_color="white", with_labels=True, font_color="black", edge_cmap=plt.cm.Blues,
    node_size=400, width=edge_widths, horizontalalignment='center', edgecolors="black", edgelist=edges, 
    edge_color=weights, arrows=False, verticalalignment='center_baseline', font_size=12
)
plt.title('Coupling Graph', fontsize=18)
plt.show()


def build_hamiltonian(coupling_matrix):
    coeffs, ops = [], []
    ns = coupling_matrix.shape[0]

    for i, j in it.combinations(range(ns), r=2):
        coeff = coupling_matrix[i, j]
        if coeff:
            for op in [qml.PauliX, qml.PauliY, qml.PauliZ]:
                coeffs.append(coeff)
                ops.append(op(i) @ op(j))

    return qml.Hamiltonian(coeffs, ops)


# build sparse hamiltonian
H = build_hamiltonian(J)
H_sparse = qml.utils.sparse_hamiltonian(H)

# diagonalize
eigvals, eigvecs = sp.sparse.linalg.eigs(H_sparse, which='SR', k=1)
eigvals = eigvals.real
ground_state = eigvecs[:, np.argmin(eigvals)]


# this circuit measures observables for the provided ground state
@qml.qnode(device=qml.device('default.qubit', wires=wires, shots=None))
def circ(observables):
    qml.QubitStateVector(ground_state, wires=range(wires))
    return [qml.expval(o) for o in observables]


def compute_exact_correlation_matrix(ground_state, wires):
    # setup observables for correlation function
    def corr_function(i, j):
        ops = []
        
        for op in [qml.PauliX, qml.PauliY, qml.PauliZ]:
            if i != j:
                ops.append(op(i) @ op(j))
            else:
                ops.append(qml.Identity(i))

        return ops
    
    # indices for sites for which correlations will be computed
    coupling_pairs = list(it.product(range(wires), repeat=2))
    
    # compute exact correlation matrix
    correlation_matrix = np.zeros((wires, wires))
    for idx, (i, j) in tqdm(enumerate(coupling_pairs), total=len(coupling_pairs)):
        observable = corr_function(i, j)

        if i == j:
            correlation_matrix[i][j] = 1.0
        else:
            correlation_matrix[i][j] = (
                    np.sum(np.array([circ(observables=[o]) for o in observable]).T) / 3
            )
            correlation_matrix[j][i] = correlation_matrix[i][j]

    return correlation_matrix


exact_correlation_matrix = compute_exact_correlation_matrix(ground_state, wires)

def compute_exact_entropy_matrix(ground_state, wires):
    ground_state_qobj = qutip.Qobj(ground_state, dims=[[2] * wires, [1] * wires])

    # compute entropies
    entropies = np.zeros(shape=(wires, wires), dtype=float)
    for i in tqdm(range(wires)):
        ptrace_diag = ground_state_qobj.ptrace(sel=[i])
        entropies[i, i] = -np.log(np.trace(ptrace_diag * ptrace_diag).real)

        for j in range(i + 1, wires):
            ptrace = ground_state_qobj.ptrace(sel=[i, j])
            e = -np.log(np.trace(ptrace * ptrace).real)
            entropies[i, j] = entropies[j, i] = e

    return entropies

exact_entropy_matrix = compute_exact_entropy_matrix(ground_state, wires)