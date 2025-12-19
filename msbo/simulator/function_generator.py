import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, List
from botorch.utils.sampling import draw_sobol_samples



class TorchDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y

    def __len__(self):
        return len(self.X)


def gen_data(
            input_dim: int,
            output_dim: int,
            n_samples: int,
            distribution: str = 'standard',
            rng: Optional[np.random._generator.Generator] = None, 
            seed: Optional[int] = None):
    if rng is None:
        rng = np.random.default_rng(seed)
    
    bounds = torch.stack([torch.zeros(input_dim), torch.ones(input_dim)], dim=0)
    X = draw_sobol_samples(bounds=bounds, n=n_samples, q=1, seed=seed).squeeze(1).numpy()
    
    if distribution=='uniform':
        y = rng.uniform(-1, 1, size=(n_samples, output_dim))
    elif distribution=='standard':
        y = rng.normal(size=(n_samples, output_dim)) 
    else:
        raise KeyError(f'{distribution} distribution not supported')
    
    # trying to push down the function close to the boundaries, so that we have less maximums on the boundaries
    for out_bound_val in [-.2, 1.2]:
        X_out = rng.uniform(size=(input_dim, input_dim))
        np.fill_diagonal(X_out, out_bound_val)
        X = np.concatenate([X, X_out], axis=0)
    y = np.concatenate([y, -1.5*np.ones(shape=(2*input_dim, output_dim))])

    dataset = TorchDataset(X, y)
    return dataset
    

class MLP(nn.Module):
    def __init__(
                self, 
                input_dim : int, 
                hidden_dims : List[int], 
                output_dim : Optional[int] = None, 
                slope : float = 0.1, 
                drop_p : float = .0):
        super().__init__()	
        layers = []
        prev_dim = input_dim
        for d in hidden_dims:
            layers.extend([nn.Linear(prev_dim, d), nn.LeakyReLU(negative_slope=slope), nn.Dropout(drop_p)])
            prev_dim = d
        if output_dim is not None:
            layers.pop() #remove last dropout
            layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.net.apply(self.init_weights)
		
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
	
    def forward(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X)
        return self.net(X.float())
        


class NN_func:
    def __init__(
                self,
                input_dim: int = 2,
                hidden_dims: List[int] = [64, 128, 32],
                output_dim: int = 1,
                n_samples: int = 10,
                squash_out: bool = False,
                distribution: str = 'standard',
                seed: Optional[int] = None,
                verbose: bool = False):
        self.verbose = verbose
        if not seed is None:
            torch.manual_seed(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = MLP(input_dim, hidden_dims, output_dim)
        self.dataset = gen_data(input_dim, output_dim, n_samples, distribution, seed=seed)
        self.train()
        self.squash_out = squash_out
        
    def train(self):
        dataloader = DataLoader(self.dataset, batch_size=16)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)#, weight_decay=.001)
        for epoch in range(800):
            self.model.train()
            cum_loss = []
            if self.verbose:
                print(f'-Epoch {epoch + 1}:')
            for X, y in dataloader:
                optimizer.zero_grad()
                pred = self.model(X)
                loss = nn.MSELoss()(pred, y)
                loss.backward()
                optimizer.step()
                cum_loss.append(loss.item())	
            if self.verbose:
                print(f"\t Train loss: {torch.tensor(cum_loss).mean()}")
        
    def sample(self, n_samples: int, seed: Optional[int] = None):
        bounds = torch.stack([torch.zeros(self.input_dim), torch.ones(self.input_dim)], dim=0)
        X = draw_sobol_samples(bounds=bounds, n=n_samples, q=1, seed=seed).squeeze(1).numpy()
        Y = self(X)
        return X, Y
        
    def __call__(self, X, grad=False):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X).float()
        if grad:
            if self.squash_out:
                out = nn.Sigmoid()(self.model(X)*.8)
            else:
                out = self.model(X)
        else:
            with torch.no_grad():
                if self.squash_out:
                    out = nn.Sigmoid()(self.model(X)*.8).numpy()
                else:
                    out = self.model(X).numpy()
        return out
    




