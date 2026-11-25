class TorchClassifierWrapper:
    """Sklearn-compatible wrapper around a PyTorch nn.Sequential routing model.

    Saves the state dict (weights) so the object is joblib-serialisable,
    and reconstructs the network architecture on each predict() call.
    """

    def __init__(self, model, input_dim, hidden_dim, output_dim, is_multiclass=True):
        self.model_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.is_multiclass = is_multiclass

    def _rebuild(self):
        import torch.nn as nn
        if self.is_multiclass:
            m = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
                nn.Softmax(dim=1),
            )
        else:
            m = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
                nn.Sigmoid(),
            )
        m.load_state_dict(self.model_weights)
        m.eval()
        return m

    def predict(self, X):
        import torch
        import numpy as np
        model = self._rebuild()
        X_arr = np.array(X, dtype=np.float32)
        single = X_arr.ndim == 1 or X_arr.shape[0] == 1
        X_tensor = torch.tensor(X_arr)
        with torch.no_grad():
            out = model(X_tensor)
        if self.is_multiclass:
            result = (torch.argmax(out, dim=1) + 1).numpy()
        else:
            result = ((out > 0.5).float() + 1).numpy().flatten().astype(int)
        # return a plain Python int for single-sample calls so int(predict(...)) works
        return int(result[0]) if single else result
