import warnings
import numpy as np
from functools import reduce

from torch import stack as torch_stack
from torch import zeros as torch_zeros
from torch import ones as torch_ones
from botorch.utils.sampling import draw_sobol_samples

from collections.abc import Callable
from typing import Any, Optional, Dict, Tuple, Union, List

from msbo.simulator.utils import at_least_2dim, split_dataset
from msbo.simulator.function_generator import NN_func
        

class MultiStepProcess:
    COMPLEX_SETTING = {'hard': (35, 51), 'simple': (3, 10), 'medium': (15, 31)}
    def __init__(
              self, 
              config: Dict,
              seed: Optional[int] = None,
              last_squashed: bool = False,
    ) -> None:
        if seed is None: 
            warnings.warn("if seed is not passed to process, results may be unreproducible")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.last_squashed = last_squashed
        self.config = config
        self._init_nodes()

    @property
    def n_processes(self) -> List[int]:
        return list(self.config['input_dim'].keys())

    def _init_nodes(self) -> None:
        self.processes = dict()
        self.process_noise = dict()
        self.measurements = dict()
        noise_config = self.config.get('noise_process', {})
        for idx in self.n_processes[:-1]:
            self.processes[idx]= self._init_process(idx)
            self.process_noise[idx] = lambda x: x + noise_config.get(idx, 0.) * self.rng.standard_normal(size=x.shape)
            self.measurements[idx] = self._init_measurement(idx)
        idx_last = self.n_processes[-1]
        self.processes[idx_last] = self._init_process(idx_last, squash_out=self.last_squashed)
        self.process_noise[idx_last] = lambda x: x + noise_config.get(idx_last, 0.) * self.rng.standard_normal(size=x.shape)
        self.measurements[idx_last] = self._init_measurement(idx_last)

    def _init_process(
            self,
            idx: int,
            squash_out: bool = True,
    ) -> Callable[[np.ndarray], np.ndarray]:
        input_dim = self.config['input_dim'][idx]
        complexity = self.config['complexity'][idx]
        if isinstance(complexity, str):
            n_samples = self.rng.integers(*self.COMPLEX_SETTING[complexity])
        elif isinstance(complexity, int):
            n_samples = complexity
        if idx>0:
            input_dim += self.config['output_dim'][idx-1] 
        return NN_func(
             input_dim=input_dim,
             output_dim=self.config['output_dim'][idx], 
             n_samples=n_samples, 
             squash_out=squash_out,
             seed=self.seed,
        )     

    def _init_measurement(
            self,
            idx: int,
    ) -> Callable[[np.ndarray], np.ndarray]:
        mode, out_dim = self.config['measurement'].get(idx, ('identity', None))
        if mode=='identity':
            m = lambda x: x
        elif mode=='filter':
            m = lambda x: x[..., :out_dim]
        else:
            raise NotImplemented
        if self.config.get('noise_meas', False):
            noise = self.config['noise_meas'].get(idx, 0.)
            add_noise = lambda x: noise*self.rng.standard_normal(size=x.shape)
            return lambda x: m(x) + add_noise(x)
        return m  
    
    def sample(
            self,
            n_samples: int,
            method: str = 'sobol',
    ) -> Dict:
        ''' 
        returns a vecDict, 
            i.e. {process_id: {'x': ndarray of shape (n_samples, dim x), 'h': same, 'm': same}, etc...}
        '''
        if method=='sobol':
            data = dict()
            tot_dim = np.sum([self.config['input_dim'][step] for step in self.n_processes])
            bounds = torch_stack([torch_zeros(tot_dim), torch_ones(tot_dim)])
            samples =  draw_sobol_samples(bounds=bounds, n=n_samples, q=1, seed=self.seed).squeeze(1).numpy()
            cumul_dim = 0
            for step in self.n_processes:
                data[step] = dict()
                d = self.config['input_dim'][step]
                if d>0:
                    data[step]['x'] =  samples[..., cumul_dim:cumul_dim+d]
                    cumul_dim += d
                else:
                    data[step]['x'] = np.empty((n_samples, d))*np.nan
        elif method=='uniform':
            data = {step: {'x': self.rng.uniform(size=(n_samples, self.config['input_dim'][step]))} for step in self.n_processes}
        else:
            raise NotImplemented
        h_list, m_list = self(data)  
        for i, k in enumerate(data.keys()):
            data[k]['h'] = h_list[i]
            data[k]['m'] = m_list[i]
        return data
    
    def call_subprocess(
            self,
            x: Optional[np.ndarray] = None,
            h: Optional[np.ndarray] = None,
            idx: int = 0,
    ):
        if idx==0:
            assert x is not None, "missing parameters"
        else:
            assert not h is None, "missing h values of previous process"
            # input of step i is concatenation of x_i and h_i-1 
            if x is None:
                x = at_least_2dim(h)
            else:
                x = np.concatenate([at_least_2dim(x), at_least_2dim(h)], axis=-1)
        h = self.processes[idx](x) 
        h = self.process_noise[idx](h)
        m = self.measurements.get(idx, lambda x: x)(h)
        return h, m
    
    def __call__(
            self, 
            data: Dict,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        ''' 
        data has to be a vecDict, 
            i.e. {process_id: {'x': ndarray of shape (n_samples, dim x), 'h': same, 'm': same}, etc...}
        '''
        to_idx = list(data.keys())[-1] + 1 ## +1 is for range()      
        from_idx = list(data.keys())[0]
        ## if first entry in data has h then start experiments from next step, otherwise from present step (only when step==0)
        previous_h = data[from_idx].get('h', None)
        if previous_h is not None:
            from_idx += 1
        h_list, m_list = [], []
        for i in range(from_idx, to_idx):
            x = data[i].get('x', None)
            h, m = self.call_subprocess(x, previous_h, idx=i)
            h_list.append(h)
            m_list.append(m)
            previous_h = h
        return h_list, m_list
        
