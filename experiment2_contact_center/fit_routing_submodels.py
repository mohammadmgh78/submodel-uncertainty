"""
Routing submodels, fitted to inputs/routing_decisions_{contact,expert}.csv.

    python fit_routing_submodels.py        (requires torch and torchbnn)

Contact side: predict the expert group from the contact group and the numbers of free experts per group.
Expert side:  predict the contact group from the expert group, the queue lengths and the waiting times.

  * frequentist: logistic regression; each instance is fitted to 80% of a bootstrap resample, and the point estimate to
    all data;
  * Bayesian: Bayesian neural network with one hidden layer of 128 nodes (torchbnn); an instance is one draw of all
    weights from the posterior, and the point estimate is the posterior mean.

The models used in the paper are in models/. The contact-side Bayesian models and the frequentist point estimates are
reproducible (the script checks this); the other instances were fitted without a seed. New models are written to
models_refit/ with fixed seeds.
"""
import ast
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import common as C
from routing_wrapper import TorchClassifierWrapper

OUT = C.path('models_refit')
HIDDEN = 128
X_CONTACT = ['contact', 'free_length_1', 'free_length_2', 'free_length_3']
X_EXPERT = ['expert', 'queue_length_1', 'queue_length_2', 'waiting_time_1', 'waiting_time_2']


def load_data():
    c = pd.read_csv(C.path('inputs', 'routing_decisions_contact.csv'))
    c[X_CONTACT[1:]] = c.pop('experts free lengths').map(ast.literal_eval).to_list()
    e = pd.read_csv(C.path('inputs', 'routing_decisions_expert.csv'))
    e[X_EXPERT[1:3]] = e.pop('contacts queue lengths').map(ast.literal_eval).to_list()
    e[X_EXPERT[3:]] = e.pop('contacts waiting times').map(ast.literal_eval).to_list()
    return c, e


def frequentist(data, name, X, y, rng):
    for k in range(C.B_MAX):
        boot = data.iloc[rng.integers(len(data), size=len(data))]
        X_train, _, y_train, _ = train_test_split(boot[X], boot[y], test_size=0.2, random_state=41)
        joblib.dump(LogisticRegression().fit(X_train, y_train), os.path.join(OUT, f'freq_{name}_{k}.pkl'))
    joblib.dump(LogisticRegression().fit(data[X], data[y]), os.path.join(OUT, f'freq_{name}_point.pkl'))


class BNN(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=d_in, out_features=HIDDEN)
        self.relu = nn.ReLU()
        self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=HIDDEN, out_features=d_out)
        self.out = nn.Softmax(dim=1) if d_out > 1 else nn.Sigmoid()

    def forward(self, x):
        return self.out(self.fc2(self.relu(self.fc1(x))))


def fixed_network(model, d_in, d_out, generator=None):
    """Network with one posterior draw of all weights (or the posterior mean if generator is None)."""
    net = nn.Sequential(nn.Linear(d_in, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, d_out),
                        nn.Softmax(dim=1) if d_out > 1 else nn.Sigmoid())

    def draw(mu, log_sigma):
        return mu if generator is None else torch.normal(mean=mu, std=torch.exp(log_sigma), generator=generator)

    with torch.no_grad():
        for layer, fc in ((net[0], model.fc1), (net[2], model.fc2)):
            layer.weight.copy_(draw(fc.weight_mu, fc.weight_log_sigma))
            layer.bias.copy_(draw(fc.bias_mu, fc.bias_log_sigma))
    return TorchClassifierWrapper(net, input_dim=d_in, hidden_dim=HIDDEN, output_dim=d_out, is_multiclass=d_out > 1)


def bayesian(data, name, X, y, d_out, split_seed):
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    X_train, _, y_train, _ = train_test_split(data[X], data[y], test_size=0.2, random_state=split_seed)
    x = torch.tensor(X_train.to_numpy(dtype=np.float32))
    if d_out > 1:
        target = torch.tensor(np.eye(d_out, dtype=np.float32)[y_train.to_numpy(dtype=int) - 1])
        criterion = nn.CrossEntropyLoss()
    else:
        target = torch.tensor((y_train.to_numpy(dtype=np.float32) - 1).reshape(-1, 1))
        criterion = nn.BCELoss()
    loader = DataLoader(TensorDataset(x, target), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(C.SEED))
    model = BNN(len(X), d_out)
    kl = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(10):
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets) + 0.01 * kl(model)
            loss.backward()
            optimizer.step()
    joblib.dump(fixed_network(model, len(X), d_out), os.path.join(OUT, f'bayes_{name}_point.pkl'))
    generator = torch.Generator().manual_seed(C.SEED + 1)
    for k in range(C.B_MAX):
        joblib.dump(fixed_network(model, len(X), d_out, generator), os.path.join(OUT, f'bayes_{name}_{k}.pkl'))


def max_difference(name):
    a, b = (joblib.load(os.path.join(d, f'{name}.pkl')) for d in (OUT, C.path('models')))
    if hasattr(a, 'coef_'):
        return max(np.abs(a.coef_ - b.coef_).max(), np.abs(a.intercept_ - b.intercept_).max())
    return max(float((a.model_weights[k] - b.model_weights[k]).abs().max()) for k in a.model_weights)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    contact, expert = load_data()
    rng = np.random.default_rng(C.SEED)
    frequentist(contact, 'contact', X_CONTACT, 'expert', rng)
    frequentist(expert, 'expert', X_EXPERT, 'contact', rng)
    bayesian(contact, 'contact', X_CONTACT, 'expert', 3, split_seed=41)
    bayesian(expert, 'expert', X_EXPERT, 'contact', 1, split_seed=42)
    for m in ['freq_contact_point', 'freq_expert_point', 'bayes_contact_point'] + [f'bayes_contact_{k}' for k in range(C.B_MAX)]:
        print(f'{m}: max difference from models/ = {max_difference(m):.1e}')
