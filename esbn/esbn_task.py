# Task shared by esbn_pnl.py and esbn_torch.py
# N_STIMULI objects are shown, the last of which repeats one of the first y_dim objects; a blank answer step
# follows, at which the model reports the position (0..y_dim-1) of the repeated object
import numpy as np

N_STIMULI = 8
y_dim = 4
T = N_STIMULI + 1  # stimuli + answer step
x_dim = 128
N_OBJECTS = 100
TRAIN_OBJECTS = np.arange(50)
TEST_OBJECTS = np.arange(50, 100)  # never seen in training


def make_objects(seed=0):
    # Random unit-length object vectors
    objects = np.random.default_rng(seed).normal(size=(N_OBJECTS, x_dim))
    return objects / np.linalg.norm(objects, axis=1, keepdims=True)


def make_sequences(objects, object_ids, n_sequences, seed):
    # Return X: (n_sequences, T, x_dim), with a blank answer step at the end, and y: (n_sequences, y_dim) one-hot
    rng = np.random.default_rng(seed)
    X = np.zeros((n_sequences, T, x_dim))
    y = np.zeros((n_sequences, y_dim))
    for i in range(n_sequences):
        # y_dim candidates followed by distractors, all distinct
        ids = rng.choice(object_ids, size=N_STIMULI - 1, replace=False)
        answer = rng.integers(y_dim)
        X[i, :N_STIMULI - 1] = objects[ids]
        X[i, N_STIMULI - 1] = objects[ids[answer]]
        y[i, answer] = 1
    return X, y
