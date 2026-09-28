# Plain-PyTorch ESBN (Webb et al., 2021, Algorithm 1), with the key-value memory held as Python lists.
# Serves as the reference that the PsyNeuLink model in esbn_pnl.py is checked against.
import numpy as np
import torch

from esbn_task import N_STIMULI, y_dim, T, x_dim, TRAIN_OBJECTS, TEST_OBJECTS, make_objects, make_sequences

# Model sizes
val_size = 32
hidden_size = 64

# Initial confidence-gate parameters
gamma_init = 1.0
beta_init = 0.0


class ESBN(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)

        def init(n_in, n_out):
            # Row-vector convention (x @ W), matching PsyNeuLink projection matrices
            return torch.nn.Parameter((torch.rand(n_in, n_out, generator=gen) * 2 - 1) / np.sqrt(n_in))

        self.gru = torch.nn.GRU(val_size + 1, hidden_size, bias=True, batch_first=True)
        self.W_y_hat = init(hidden_size, y_dim)
        self.W_g = init(hidden_size, 1)
        self.W_v_w = init(hidden_size, val_size)
        self.gamma = torch.nn.Parameter(torch.tensor(gamma_init))
        self.beta = torch.nn.Parameter(torch.tensor(beta_init))

    def forward(self, X):
        # X: (batch, T, x_dim); returns y_hat logits at every step: (batch, T, y_dim)
        batch = X.shape[0]
        keys, values = [], []  # esbn_keys: embeddings z; esbn_vals: written values v_w
        v_r = torch.zeros(batch, val_size + 1)
        h = torch.zeros(1, batch, hidden_size)
        y_hats = []
        for t in range(X.shape[1]):
            z = X[:, t]
            out, h = self.gru(v_r[:, None], h)
            h_t = out[:, 0]
            y_hats.append(h_t @ self.W_y_hat)
            g = torch.sigmoid(h_t @ self.W_g)
            v_w = torch.relu(h_t @ self.W_v_w)
            if keys:
                M_k = torch.stack(keys, 1)                        # (batch, t, x_dim)
                M_v = torch.stack(values, 1)                      # (batch, t, val_size)
                dot_product = (M_k @ z[:, :, None])[:, :, 0]      # (batch, t)
                w = torch.softmax(dot_product, 1)
                c = torch.sigmoid(self.gamma * dot_product + self.beta)
                v_r = g * torch.cat([(w[:, :, None] * M_v).sum(1), (w * c).sum(1, keepdim=True)], 1)
            else:
                v_r = torch.zeros(batch, val_size + 1)
            keys.append(z)
            values.append(v_w)
        return torch.stack(y_hats, 1)


def accuracy(model, X, y):
    with torch.no_grad():
        y_hat = model(torch.tensor(X))[:, -1]
    return float((y_hat.argmax(1).numpy() == y.argmax(1)).mean())


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    objects = make_objects()
    X_train, y_train = make_sequences(objects, TRAIN_OBJECTS, 2000, seed=1)
    X_test, y_test = make_sequences(objects, TEST_OBJECTS, 500, seed=2)
    model = ESBN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = 16
    for epoch in range(5):
        for i in range(0, len(X_train), batch):
            y_hat = model(torch.tensor(X_train[i:i + batch]))[:, -1]
            loss = torch.nn.functional.cross_entropy(y_hat, torch.tensor(y_train[i:i + batch]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'epoch {epoch}: loss {loss.item():.3f}  train acc {accuracy(model, X_train[:500], y_train[:500]):.2f}'
              f'  test acc (novel objects) {accuracy(model, X_test, y_test):.2f}')
