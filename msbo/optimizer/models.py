import torch
from torch import Tensor
from botorch.models.model import Model
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch import fit_gpytorch_model
from botorch.posteriors import Posterior
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel

from typing import Dict, List, Any, Optional
from msbo.optimizer.utils import at_least_2dim


class CascadeNormal(Posterior):
    def __init__(self, GPs, dataset, X):
        self._GPs = GPs
        self._dataset = dataset
        self.X = X

    @property
    def device(self) -> torch.device:
        r"""Torch device of the posterior."""
        return self.X.device 

    @property
    def dtype(self) -> torch.dtype:
        r"""Torch dtype of the posterior."""
        return torch.double

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample) of the posterior."""
        return torch.Size([self.X.shape[0], self._GPs[-1].num_outputs])
    
    def rsample(
            self,
            sample_shape: torch.Size = torch.Size(), 
            base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        # x <- x_t + x_t-1 + m_t-1
        current_dim = 0
        for GP, process_id in zip(self._GPs, self._dataset.keys()):
            dataset = self._dataset[process_id]
            if process_id==0:
                dim = dataset['x'].shape[-1]
                x_curr = at_least_2dim(self.X[..., :dataset['tunable']])
                x = at_least_2dim(self.X[..., :dim])
                posterior_i = GP.posterior(x)
                if base_samples is not None:
                    samples_i = posterior_i.rsample(sample_shape, base_samples.unsqueeze(-1))
                else:
                    samples_i = posterior_i.rsample(sample_shape)
            else:
                dim = dataset['tunable']
                x_curr = at_least_2dim(self.X[..., current_dim:current_dim+dim])
                aux_shape = [sample_shape[0]] + [1] * x_curr.ndim
                if dataset['residual']:
                    samples_i = torch.cat([x_prev.repeat(*aux_shape), samples_i], dim=-1)
                x = torch.cat([x_curr.repeat(*aux_shape), samples_i], dim=-1)
                posterior_i = GP.posterior(x)  #GP.posteriors has a log_prob function
                samples_i = posterior_i.rsample().squeeze(0)
            x_prev = x_curr
            current_dim += dim
        return samples_i.to(self.X)
    
    def sample(self, n_samples, sample_shape: torch.Size = torch.Size(), base_samples: Optional[Tensor] = None) -> Tensor:
        sample_shape = torch.Size([n_samples])
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, base_samples=base_samples)


class Cascade_GP(Model):
    '''
    TODO 
        add normalization for inputs coming from precedent GPs
        add posterior given first step input, marginalizing over second step parameters
    '''
    def __init__(
            self, 
            dataset, 
            GPs: Optional[List] = None,
            mlls: Optional[List] = None,
        ) -> None:
        super().__init__()
        self.dataset = dataset
        if GPs is None:
            self.GPs = []
            self.mlls = []
            self._initialize_GPs()
        else:
            self.GPs = GPs
            self.mlls = mlls
    
    def _initialize_GPs(self):
        for k, data in self.dataset.items():
            xtrain, ytrain = data['x'], data['y']
            # print(f'x:{xtrain.shape}, y:{ytrain.shape}')
            model, mll = self._get_model_and_mll(xtrain, ytrain)
            self.GPs.append(model)
            self.mlls.append(mll)

    def _get_model_and_mll(self, xtrain, ytrain):   
        models = []
        for i in range(ytrain.shape[-1]):
            models.append(
                SingleTaskGP(
                    xtrain,
                    ytrain[..., i:i+1],
                    covar_module=ScaleKernel(
                        RBFKernel(ard_num_dims=xtrain.shape[-1])
                    ),
                    outcome_transform=Standardize(m=1),
                    input_transform=Normalize(d=xtrain.shape[-1]),
                )
            )
        model = ModelListGP(*models).to(xtrain)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return model, mll

    def fit(self):
        for mll in self.mlls:
            fit_gpytorch_model(mll)

    @property
    def num_outputs(self):
        return self.GPs[-1].num_outputs
  
    def condition_on_observations(self) -> Model:
        '''
        get conditioned model on previous process steps
        '''
        pass

    def extract_subsequence(self, idx=0) -> Model:
        GPs = self.GPs[idx:]
        mlls = self.mlls[idx:]
        dataset = {k-idx: v for k,v in self.dataset.items() if k>=idx}
        subsequence = Cascade_GP(
            dataset=dataset, 
            GPs=GPs,
            mlls=mlls,
        )
        return subsequence

    def posterior(self, X: Tensor, **kwargs) -> CascadeNormal:
        return CascadeNormal(self.GPs, self.dataset, X)

    def forward(self, X: Tensor) -> CascadeNormal:
        '''
        If it is a subsequence (not starting from root node), 
        the X has to have also the previous step m (and x if necessary) already concatenated 
        (since x = x_t + x_t-1, m_t-1)
        '''
        return self.posterior(X)
    
