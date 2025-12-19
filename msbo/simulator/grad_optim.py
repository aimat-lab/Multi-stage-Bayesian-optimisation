import torch
import numpy as np
import torch.nn as nn

from typing import Callable, Optional, Tuple
from scipy.optimize import minimize, Bounds
from botorch.utils.sampling import draw_sobol_samples


class DifferentiableProcess(nn.Module):
    def __init__(self, process) -> None:
        super().__init__()
        self.input_dim = list(process.config['input_dim'].values())
        self.processes = process.processes
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        lower = 0
        out = torch.tensor([])
        for step, input_dim in enumerate(self.input_dim):
            upper = lower + input_dim
            x = torch.concat([X[..., lower:upper], out], dim=-1)
            out = self.processes[step](x, grad=True)
            lower = upper
        return out


def _arrayify(X: torch.Tensor) -> np.ndarray:
    return X.cpu().detach().contiguous().double().clone().numpy()


def get_maximum(
        process, 
        seed: Optional[int] = None,
        raw_samples: int = 512,
        batch_limit: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    diff_process = DifferentiableProcess(process)
    def f(x):
        return -diff_process(x)

    tot_dim = np.sum([process.config['input_dim'][step] for step in process.n_processes])
    bounds = torch.stack([torch.zeros(tot_dim), torch.ones(tot_dim)])
    initial_conditions = draw_sobol_samples(bounds=bounds, n=raw_samples, q=1, seed=seed).squeeze(1)

    batched_initial_conditions = initial_conditions.split(batch_limit)
    candidates_list, values_list = [], []
    for i, initial_conditions in enumerate(batched_initial_conditions):
        shapeX = initial_conditions.shape
        
        def f_np_wrapper(x: np.ndarray, f: Callable):
            """Given a torch callable, compute value + grad given a numpy array."""
            if np.isnan(x).any():
                raise RuntimeError(
                    f"{np.isnan(x).sum()} elements of the {x.size} element array "
                    f"`x` are NaN."
                )
            X = (
                torch.from_numpy(x)
                .to(initial_conditions)
                .view(shapeX)
                .contiguous()
                .requires_grad_(True)
            )
            loss = f(X).sum()
            # compute gradient w.r.t. the inputs (does not accumulate in leaves)
            gradf = _arrayify(torch.autograd.grad(loss, X)[0].contiguous().view(-1))
            if np.isnan(gradf).any():
                msg = (
                    f"{np.isnan(gradf).sum()} elements of the {x.size} element "
                    "gradient array `gradf` are NaN. This often indicates numerical issues."
                )
                if initial_conditions.dtype != torch.double:
                    msg += " Consider using `dtype=torch.double`."
                raise RuntimeError(msg)
            fval = loss.item()
            return fval, gradf
    
        x0 = _arrayify(initial_conditions.view(-1))
        res = minimize(
            fun=f_np_wrapper,
            args=(f,),
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            bounds=Bounds(lb=0., ub=1.),
            options={"maxiter": 200},
        )
        if "success" not in res.keys() or "status" not in res.keys():
            print(f'Optimization {i} failed')

        candidates = torch.from_numpy(res.x).to(initial_conditions).reshape(shapeX)
        candidates_list.append(candidates)
        values_list.append(diff_process(candidates))

    candidates = torch.cat(candidates_list, dim=0)
    values = torch.cat(values_list, dim=0)
    best_idx = values.argmax().item()
    return _arrayify(candidates[best_idx]), _arrayify(values[best_idx])


