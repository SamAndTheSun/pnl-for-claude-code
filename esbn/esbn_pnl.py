# Paths and imports
import copy
import warnings
import numpy as np
import torch
from psyneulink import *

from esbn_task import N_STIMULI, y_dim, T, x_dim, TRAIN_OBJECTS, TEST_OBJECTS, make_objects, make_sequences

torch.set_default_dtype(torch.float64)

# Overall structure (one pass per step of a sequence; the whole sequence is one trial, trained with
# backpropagation through time):
#
#                 y_hat ----> loss <---- y_target
#                   ^
#                   | gru_to_y_hat
#                   |
#            +---> gru ------------+---------------+
#            |      |              |               |
#            |      | gru_to_g     | gru_to_v_w    |
#            |      v              v               |
#            |      g            v_w               |
#            |      |              |               |
#  v_r_to_gru|      |              v               |
#            |      |          esbn_vals <---> vals_delay
#            |      |              |
#            +--- v_r_gen <--------+
#                  ^    ^
#           c_gate |    | attn_gate
#                  |    |
#              c_gate  attn_gate
#                  ^    ^
#                  |    | dot_product
#                  +----+
#                    |
#                esbn_keys <---> keys_delay
#                    ^
#                    |
#                  embed
#
# c_gate in detail:
#
#                            c_gate <------------------------+
#                               ^                            |
#                               | scaled_to_c_gate           | beta_broadcast
#                               |                            |
#                       scaled_dot_product             beta_endpoint
#                        ^              ^                    ^
#  dot_product_to_scaled |              | gamma_broadcast    | beta
#                        |              |                    |
#                        |       gamma_endpoint              |
#                        |              ^                    |
#                        |              | gamma              |
#                        |              |                    |
#                        |          const_one ---------------+
#                        |
#                    esbn_keys
#
# The slot input (one-hot position of the current stimulus, all zeros on the answer step) also feeds
# esbn_keys, esbn_vals, attn_gate and v_r_gen.
# For any unlabelled edge, just assume it sends the full output (e.g. for embed, it just sends the embed)

# Set hyperparameters
gamma_init = 1.0
beta_init = 0.0
val_size = 32     # dim of the write value, outputted by the GRU
hidden_size = 64  # dim of the GRU's hidden state
learning_rate = 1e-3
batch_size = 16

# Initial values of the learnable matrices (same convention as PsyNeuLink: sender value @ matrix)
rng = np.random.default_rng(0)
def init_matrix(n_in, n_out):
    return rng.uniform(-1, 1, (n_in, n_out)) / np.sqrt(n_in)

# The slot input for every step of a sequence: one-hot for each stimulus, all zeros for the answer step
SLOTS = np.vstack([np.eye(N_STIMULI), np.zeros(N_STIMULI)])


# ===== Packing helpers ===== #

# In PyTorch mode, a node whose input ports differ in size has its function run once per port, so a node that
# combines several inputs takes them through a single input port, each sender placed in its own slice.
# place() is the matrix that writes a sender's value into the slice starting at offset
def place(n_in, n_out, offset):
    matrix = np.zeros((n_in, n_out))
    matrix[:, offset:offset + n_in] = np.eye(n_in)
    return matrix

# select() is the matrix that reads the slice of length n starting at offset
def select(n_in, offset, n):
    matrix = np.zeros((n_in, n))
    matrix[offset:offset + n] = np.eye(n)
    return matrix

# Create a node whose function is written once in torch; the numpy version (used by PsyNeuLink to set up the node)
# calls the same torch function
def torch_node(name, n_in, torch_function):
    def numpy_function(variable):
        return torch_function(torch.as_tensor(np.asarray(variable, dtype=float))).numpy()
    return ProcessingMechanism(name=name, input_shapes=n_in,
                               function=UserDefinedFunction(numpy_function, default_variable=np.zeros(n_in),
                                                            pytorch_function_generator=lambda device, context:
                                                            torch_function))

# The answer step is the step whose slot is all zeros
# On it, the memory nodes and v_r_gen output new zero tensors, which empties the memories and v_r for the first
# step of the next sequence; being new tensors, they also stop gradients flowing back into the previous sequence
def is_answer_step(slot_value):
    return bool((slot_value.sum(-1) == 0).all())

# Write item into its slot of a flattened memory
def write_memory(memory, slot_value, item):
    return memory + (slot_value[..., :, None] * item[..., None, :]).flatten(-2)


# ===== Input nodes ===== #

# Create the embed node, which receives the 1d embedding as input
embed = ProcessingMechanism(name='embed', input_shapes=x_dim)

# Create the slot node, which receives the one-hot position of the current stimulus
slot = ProcessingMechanism(name='slot', input_shapes=N_STIMULI)

# Create the y_target node, which receives the correct answer (only its value on the answer step is used)
y_target = ProcessingMechanism(name='y_target', input_shapes=y_dim)


# ===== esbn_keys infrastructure ===== #

# esbn_keys' single input port holds [stored keys (N_STIMULI * x_dim) | slot (N_STIMULI) | embed (x_dim)]
KEYS_MEMORY = N_STIMULI * x_dim
KEYS_IN = KEYS_MEMORY + N_STIMULI + x_dim

# esbn_keys dot products embed with every stored key, then stores embed in its slot
# It outputs [stored keys | dot_product]; slots without a key have a dot product of 0
def esbn_keys_function(variable):
    stored, slot_value, query = torch.split(variable, [KEYS_MEMORY, N_STIMULI, x_dim], dim=-1)
    if is_answer_step(slot_value):
        return torch.zeros(*variable.shape[:-1], KEYS_MEMORY + N_STIMULI)
    dot_product = (stored.unflatten(-1, (N_STIMULI, x_dim)) @ query[..., :, None])[..., 0]
    return torch.cat([write_memory(stored, slot_value, query), dot_product], dim=-1)

esbn_keys = torch_node('esbn_keys', KEYS_IN, esbn_keys_function)

# keys_delay holds the stored keys until the next step, where they return to esbn_keys
# (a projection from a node to itself is dropped in PyTorch mode, so the loop goes through a second node)
# It is the size of esbn_keys' input, with the stored keys already in their slice: before its first execution,
# a feedback projection sends a default the size of its receiver through its matrix, so it must be square
keys_delay = ProcessingMechanism(name='keys_delay', input_shapes=KEYS_IN)

# Send the stored keys to keys_delay, into their slice of esbn_keys' input
keys_to_delay = MappingProjection(name='keys_to_delay', sender=esbn_keys, receiver=keys_delay,
                                  matrix=select(KEYS_MEMORY + N_STIMULI, 0, KEYS_MEMORY)
                                  @ place(KEYS_MEMORY, KEYS_IN, 0),
                                  learnable=False)

# Send the stored keys back to esbn_keys on the next step
delay_to_keys = MappingProjection(name='delay_to_keys', sender=keys_delay, receiver=esbn_keys,
                                  matrix=np.eye(KEYS_IN), learnable=False, feedback=True)

# Send the slot and the embedding to esbn_keys
slot_to_keys = MappingProjection(name='slot_to_keys', sender=slot, receiver=esbn_keys,
                                 matrix=place(N_STIMULI, KEYS_IN, KEYS_MEMORY), learnable=False)
embed_to_keys = MappingProjection(name='embed_to_keys', sender=embed, receiver=esbn_keys,
                                  matrix=place(x_dim, KEYS_IN, KEYS_MEMORY + N_STIMULI), learnable=False)


# ===== c_gate infrastructure ===== #

# STEP numbers are conceptual moves in the flow of information, not the order of the code:
#   STEP 1: const_one                        (node)        outputs 1
#   STEP 2: gamma, beta                      (projections) the learned scalars: 1 -> gamma_endpoint, beta_endpoint
#   STEP 3: gamma_endpoint, beta_endpoint    (nodes)       hold the learned scalars
#   STEP 4: gamma_broadcast, beta_broadcast  (projections) broadcast the scalars to the dot product's shape
#   STEP 5: dot_product_to_scaled            (projection)  bring in the dot products from esbn_keys
#   STEP 6: scaled_dot_product               (node)        gamma * dot_product
#   STEP 7: scaled_to_c_gate                 (projection)  send gamma * dot_product to c_gate
#   STEP 8: c_gate                           (node)        sigmoid(gamma * dot_product + beta)
# Nodes are defined first because each projection needs its sender and receiver to already exist

# ----- nodes ----- #

# Define a node that always outputs 1, without needing any input
# Nodes are not learnable in PsyNeuLink, so this constant drives the learnable weights gamma and beta
# default_input=DEFAULT_VARIABLE makes it use its default_variable (1) as its input, so it acts as a BIAS node
# It has to be given as an InputPort: DEFAULT_INPUT in a port specification dict is silently ignored
const_one = ProcessingMechanism(name='const_one', default_variable=[1.0],
                                input_ports=[InputPort(name='IN', default_input=DEFAULT_VARIABLE)])

# Define the endpoints of the learnable weights gamma and beta
# Since const_one outputs 1, their value = 1 * learned weight, i.e. the current value of gamma and beta
gamma_endpoint = ProcessingMechanism(name='gamma_endpoint', input_shapes=1)
beta_endpoint = ProcessingMechanism(name='beta_endpoint', input_shapes=1)

# Define scaled_dot_product, which multiplies the dot products by gamma
# It has two ports: one for the dot products and one for gamma broadcast to every slot.
# PRODUCT then multiplies the two ports elementwise: DOT_PRODUCT * GAMMA
scaled_dot_product = ProcessingMechanism(
    name = 'scaled_dot_product',
    input_ports=[{NAME: 'DOT_PRODUCT', VARIABLE: np.zeros(N_STIMULI)},
                 {NAME: 'GAMMA',       VARIABLE: np.zeros(N_STIMULI)}],
    function=LinearCombination(operation=PRODUCT))

# Define c_gate, which adds beta to the scaled dot products and then applies the sigmoid
# It has a single input port, which sums everything projecting to it (scaled_dot_product + beta broadcast to
# every slot); Logistic then applies the sigmoid to that sum
c_gate = ProcessingMechanism(name='c_gate', input_shapes=N_STIMULI, function=Logistic())

# ----- projections ----- #

# Define the learned scalars gamma and beta, the only learnable weights here
# Each is a 1x1 weight from const_one, so it learns a single value shared by every slot
gamma = MappingProjection(name='gamma', sender=const_one, receiver=gamma_endpoint,
                          matrix=[[gamma_init]])
beta = MappingProjection(name='beta', sender=const_one, receiver=beta_endpoint,
                         matrix=[[beta_init]])

# Broadcast gamma to the same shape as the dot products ([γ] -> [γ, γ, ..., γ]) and send it to
# scaled_dot_product's GAMMA port
# Fixed (not learnable), so every slot keeps the same gamma; if learnable, each slot would learn its own weight
gamma_broadcast = MappingProjection(name='gamma_broadcast', sender=gamma_endpoint,
                                    receiver=scaled_dot_product.input_ports['GAMMA'],
                                    matrix=np.ones((1, N_STIMULI)), learnable=False)

# Broadcast beta to the same shape as the dot products ([β] -> [β, β, ..., β]) and send it to c_gate,
# whose input port adds it to the scaled dot products
# Fixed (not learnable), so every slot keeps the same beta
beta_broadcast = MappingProjection(name='beta_broadcast', sender=beta_endpoint, receiver=c_gate,
                                   matrix=np.ones((1, N_STIMULI)), learnable=False)

# Send the dot products from esbn_keys (0 for empty slots) to scaled_dot_product's DOT_PRODUCT port
# Fixed, so they arrive unchanged
dot_product_to_scaled = MappingProjection(name='dot_product_to_scaled', sender=esbn_keys,
                                          receiver=scaled_dot_product.input_ports['DOT_PRODUCT'],
                                          matrix=select(KEYS_MEMORY + N_STIMULI, KEYS_MEMORY, N_STIMULI),
                                          learnable=False)

# Send gamma * dot_product to c_gate, where its input port adds beta
# Fixed identity matrix, so it arrives unchanged
scaled_to_c_gate = MappingProjection(name='scaled_to_c_gate', sender=scaled_dot_product, receiver=c_gate,
                                     matrix=np.eye(N_STIMULI), learnable=False)


# ===== attn_gate infrastructure ===== #

# attn_gate's single input port holds [dot_product (N_STIMULI) | empty (N_STIMULI)], where empty is 1 for
# slots that don't hold a key yet
# attn_gate is the softmax of the dot products over the stored keys only; empty slots get exactly 0,
# and before any key is stored every weight is 0
def attn_gate_function(variable):
    dot_product, empty = torch.split(variable, [N_STIMULI, N_STIMULI], dim=-1)
    exp = torch.exp(dot_product - dot_product.max(-1, keepdim=True).values) * (1 - empty)
    total = exp.sum(-1, keepdim=True)
    return exp / torch.where(total > 0, total, torch.ones_like(total))

attn_gate = torch_node('attn_gate', 2 * N_STIMULI, attn_gate_function)

# Send the dot products from esbn_keys to attn_gate
dot_product_to_attn = MappingProjection(name='dot_product_to_attn', sender=esbn_keys, receiver=attn_gate,
                                        matrix=select(KEYS_MEMORY + N_STIMULI, KEYS_MEMORY, N_STIMULI)
                                        @ place(N_STIMULI, 2 * N_STIMULI, 0),
                                        learnable=False)

# Send which slots are empty to attn_gate: slot i and every later slot are empty on step i,
# and on the answer step (slot all zeros) none are
slot_to_attn = MappingProjection(name='slot_to_attn', sender=slot, receiver=attn_gate,
                                 matrix=np.triu(np.ones((N_STIMULI, N_STIMULI))) @ place(N_STIMULI, 2 * N_STIMULI,
                                                                                          N_STIMULI),
                                 learnable=False)


# ===== GRU infrastructure ===== #

# Define the GRU controller; its input is v_r (the retrieved value plus its confidence) from the previous step
gru = GRUComposition(name='gru', input_size=val_size + 1, hidden_size=hidden_size, bias=True)

# Define v_r_hold, which holds v_r from the previous step and passes it to the GRU
# (a feedback projection straight into the nested GRU breaks PsyNeuLink's feedback bookkeeping, so the
# feedback projection ends here instead)
v_r_hold = ProcessingMechanism(name='v_r_hold', input_shapes=val_size + 1)
v_r_hold_to_gru = MappingProjection(name='v_r_hold_to_gru', sender=v_r_hold, receiver=gru.input_node,
                                    matrix=np.eye(val_size + 1), learnable=False)

# Define the GRU's outputs: the answer, the retrieval gate, and the value written to esbn_vals
y_hat = ProcessingMechanism(name='y_hat', input_shapes=y_dim)
g = ProcessingMechanism(name='g', input_shapes=1, function=Logistic())
v_w = ProcessingMechanism(name='v_w', input_shapes=val_size, function=ReLU())

# Define the learned output layers, each a full weight matrix from the GRU's hidden state
gru_to_y_hat = MappingProjection(name='gru_to_y_hat', sender=gru.output_node, receiver=y_hat,
                                 matrix=init_matrix(hidden_size, y_dim), learnable=True)
gru_to_g = MappingProjection(name='gru_to_g', sender=gru.output_node, receiver=g,
                             matrix=init_matrix(hidden_size, 1), learnable=True)
gru_to_v_w = MappingProjection(name='gru_to_v_w', sender=gru.output_node, receiver=v_w,
                               matrix=init_matrix(hidden_size, val_size), learnable=True)


# ===== esbn_vals infrastructure ===== #

# esbn_vals' single input port holds [stored values (N_STIMULI * val_size) | slot (N_STIMULI) | v_w (val_size)]
VALS_MEMORY = N_STIMULI * val_size
VALS_IN = VALS_MEMORY + N_STIMULI + val_size

# esbn_vals stores v_w in its slot and outputs the stored values
def esbn_vals_function(variable):
    stored, slot_value, value = torch.split(variable, [VALS_MEMORY, N_STIMULI, val_size], dim=-1)
    if is_answer_step(slot_value):
        return torch.zeros(*variable.shape[:-1], VALS_MEMORY)
    return write_memory(stored, slot_value, value)

esbn_vals = torch_node('esbn_vals', VALS_IN, esbn_vals_function)

# vals_delay holds the stored values until the next step, where they return to esbn_vals
# Like keys_delay, it is the size of esbn_vals' input, with the stored values already in their slice
vals_delay = ProcessingMechanism(name='vals_delay', input_shapes=VALS_IN)
vals_to_delay = MappingProjection(name='vals_to_delay', sender=esbn_vals, receiver=vals_delay,
                                  matrix=place(VALS_MEMORY, VALS_IN, 0), learnable=False)
delay_to_vals = MappingProjection(name='delay_to_vals', sender=vals_delay, receiver=esbn_vals,
                                  matrix=np.eye(VALS_IN), learnable=False, feedback=True)

# Send the slot and v_w to esbn_vals
slot_to_vals = MappingProjection(name='slot_to_vals', sender=slot, receiver=esbn_vals,
                                 matrix=place(N_STIMULI, VALS_IN, VALS_MEMORY), learnable=False)
v_w_to_vals = MappingProjection(name='v_w_to_vals', sender=v_w, receiver=esbn_vals,
                                matrix=place(val_size, VALS_IN, VALS_MEMORY + N_STIMULI), learnable=False)


# ===== v_r_gen infrastructure ===== #

# v_r_gen's single input port holds
# [stored values (N_STIMULI * val_size) | attn_gate (N_STIMULI) | c_gate (N_STIMULI) | g (1) | slot (N_STIMULI)]
V_R_IN = VALS_MEMORY + 3 * N_STIMULI + 1

# v_r_gen outputs v_r = g * [attention-weighted stored values, attention-weighted confidence]
def v_r_gen_function(variable):
    stored, attn, conf, gate, slot_value = torch.split(variable, [VALS_MEMORY, N_STIMULI, N_STIMULI, 1, N_STIMULI],
                                                       dim=-1)
    if is_answer_step(slot_value):
        return torch.zeros(*variable.shape[:-1], val_size + 1)
    values = stored.unflatten(-1, (N_STIMULI, val_size))
    retrieved = (attn[..., None, :] @ values)[..., 0, :]
    confidence = (attn * conf).sum(-1, keepdim=True)
    return gate * torch.cat([retrieved, confidence], dim=-1)

v_r_gen = torch_node('v_r_gen', V_R_IN, v_r_gen_function)

# Send its inputs to v_r_gen, each into its slice
vals_to_v_r = MappingProjection(name='vals_to_v_r', sender=esbn_vals, receiver=v_r_gen,
                                matrix=place(VALS_MEMORY, V_R_IN, 0), learnable=False)
attn_to_v_r = MappingProjection(name='attn_to_v_r', sender=attn_gate, receiver=v_r_gen,
                                matrix=place(N_STIMULI, V_R_IN, VALS_MEMORY), learnable=False)
c_gate_to_v_r = MappingProjection(name='c_gate_to_v_r', sender=c_gate, receiver=v_r_gen,
                                  matrix=place(N_STIMULI, V_R_IN, VALS_MEMORY + N_STIMULI), learnable=False)
g_to_v_r = MappingProjection(name='g_to_v_r', sender=g, receiver=v_r_gen,
                             matrix=place(1, V_R_IN, VALS_MEMORY + 2 * N_STIMULI), learnable=False)
slot_to_v_r = MappingProjection(name='slot_to_v_r', sender=slot, receiver=v_r_gen,
                                matrix=place(N_STIMULI, V_R_IN, VALS_MEMORY + 2 * N_STIMULI + 1), learnable=False)

# Send v_r to v_r_hold unchanged; it is a feedback projection, so v_r_hold receives it on the next step
v_r_to_hold = MappingProjection(name='v_r_to_hold', sender=v_r_gen, receiver=v_r_hold,
                                matrix=np.eye(val_size + 1), learnable=False, feedback=True)


# ===== Composition ===== #

esbn = AutodiffComposition(name='esbn', full_sequence_mode=True, loss_spec=Loss.CROSS_ENTROPY,
                           optimizer_type='adam', learning_rate=learning_rate, targets=(y_hat, y_target))
esbn.add_nodes([embed, slot, y_target, esbn_keys, keys_delay, const_one, gamma_endpoint, beta_endpoint,
                scaled_dot_product, c_gate, attn_gate, v_r_hold, gru, y_hat, g, v_w, esbn_vals, vals_delay, v_r_gen])
for projection in [keys_to_delay, delay_to_keys, slot_to_keys, embed_to_keys,
                   gamma, beta, gamma_broadcast, beta_broadcast, dot_product_to_scaled, scaled_to_c_gate,
                   dot_product_to_attn, slot_to_attn,
                   v_r_hold_to_gru, gru_to_y_hat, gru_to_g, gru_to_v_w,
                   vals_to_delay, delay_to_vals, slot_to_vals, v_w_to_vals,
                   vals_to_v_r, attn_to_v_r, c_gate_to_v_r, g_to_v_r, slot_to_v_r, v_r_to_hold]:
    esbn.add_projection(projection, feedback=projection.feedback)

# v_r_hold's only input is a feedback projection, and the delay nodes' and v_r_gen's only outputs are
# feedback projections, so they would otherwise be treated as INPUT and OUTPUT nodes
esbn.exclude_node_roles(v_r_hold, NodeRole.INPUT)
for node in [keys_delay, vals_delay, v_r_gen]:
    esbn.exclude_node_roles(node, NodeRole.OUTPUT)


# ===== Training and testing ===== #

# One trial per sequence: every input node gets the whole sequence
def sequence_inputs(X, y):
    return [{embed: X[i].tolist(), slot: SLOTS.tolist(), y_target: [y[i].tolist()] * T} for i in range(len(X))]

# Return the answer-step y_hat for each sequence of the last call to learn()
def answers(n_sequences):
    return np.array([np.asarray(r[0], dtype=float).reshape(-1) for r in esbn.results[-n_sequences:]])

# Node values are not copied back from PyTorch: copying the GRU's fails for batches of more than one sequence,
# and the GRU starts each sequence from its node value, which must stay at its initial zeros
def learn(X, y, epochs=1, call_before_minibatch=None):
    esbn.learn(inputs=sequence_inputs(X, y), epochs=epochs, minibatch_size=batch_size,
               execution_mode=ExecutionMode.PyTorch, synch_node_values_with_torch=None,
               call_before_minibatch=call_before_minibatch)

def train(X, y, epochs=1):
    learn(X, y, epochs)

# Test by running the sequences through learn() with the learning rate set to 0; the optimizer's state is saved
# before the first minibatch and restored afterwards, so testing doesn't affect training
# (learn()'s own learning_rate argument isn't used: it stays applied to later calls)
def test(X, y):
    saved = []
    def freeze():
        optimizer = esbn.pytorch_representation.optimizer
        if not saved:
            saved.append(copy.deepcopy(optimizer.state_dict()))
        for group in optimizer.param_groups:
            group['lr'] = 0.0
    learn(X, y, call_before_minibatch=freeze)
    esbn.pytorch_representation.optimizer.load_state_dict(saved[0])
    return float((answers(len(X)).argmax(1) == y.argmax(1)).mean())


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    objects = make_objects()
    X_train, y_train = make_sequences(objects, TRAIN_OBJECTS, 2000, seed=1)
    X_test, y_test = make_sequences(objects, TEST_OBJECTS, 500, seed=2)
    for epoch in range(5):
        train(X_train, y_train)
        print(f'epoch {epoch}: train acc {test(X_train[:500], y_train[:500]):.2f}'
              f'  test acc (novel objects) {test(X_test, y_test):.2f}')
