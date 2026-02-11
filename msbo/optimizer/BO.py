import torch
import numpy as np
from typing import List, Optional

from msbo.optimizer.models import Cascade_GP
from msbo.optimizer.lab_interface import get_dataset_from_platform
from msbo.optimizer.acquisition_function import generate_acqf_dict, cascade_optimize_acqf, _init_acqf
from msbo.optimizer.utils import OptimizationSummary
from msbo.simulator.grad_optim import DifferentiableProcess

from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf



class BayesOptimizer:
    def __init__(
            self,
            oracle, 
            q: int = 1, 
            name: str = '',
            obj_idx: int = -1,
            save_path: str = './',
            duplicate_samples: bool = False,
            cost_aware: bool = False,
            global_optimization: bool = False,
            fixed_xs: bool = False,
    ):
        '''only sequential at the moment (q must be 1!), needs to be extended for batched experiments'''
        self.tkwargs = {
                "dtype": torch.double,
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            }
        # self.summary = OptimizationSummary(process, name=name)
        self.global_optimization = global_optimization
        self.q = q
        self.obj_idx = 0 if global_optimization else oracle.process.n_processes[obj_idx]
        self.oracle = oracle
        self.name = name
        self.save_path = save_path
        self.summary = OptimizationSummary()
        self.duplicate_samples = duplicate_samples
        self.cost_aware = cost_aware
        self.fixed_xs = fixed_xs

    def _update_dataset(self):
        self.dataset = get_dataset_from_platform(self.oracle, tkwargs=self.tkwargs, msbo=not self.global_optimization)
        value, idx = torch.max(self.dataset[self.obj_idx]['y'], 0)
        self.best_observed_value, self.best_sample = value.item(), self.dataset[self.obj_idx]['ids'][idx].int().item()
    
    def generate_initial_data(
            self, 
            n: int = 10, 
            method: str ='sobol',
    ):
        self.oracle.randomized_sampling(n_samples=n, method=method)
        self._update_dataset()
    
    def initialize_model(self, state_dict=None):
        self.model = Cascade_GP(self.dataset)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)    

    def global_step(self, acqf):
        n = sum([data['tunable'] for _, data in self.dataset.items()])
        bounds = torch.tensor([[0.0] * n, [1.0] * n], **self.tkwargs)
        sampled_subseq = 0    
        sampled_id = None
        candidates, acqf_values = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=self.q,
            num_restarts=10,
            raw_samples=512,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        ) 
        return candidates, acqf_values, sampled_subseq, sampled_id
    
    def msbo_step(self, acqf_dict, subseq=None):
        ## create list of bounds for each subsequence of processes
        tunable_dims = []
        input_dims = []
        for _, summary in self.model.dataset.items():
            tunable_dims.append(summary['tunable'])
            input_dims.append(summary['x'].shape[-1])
        bounds_list = []
        for i in range(len(tunable_dims)):
            n = sum(tunable_dims[i+1:]) + input_dims[i]
            bounds_list.append(torch.tensor([[0.0] * n, [1.0] * n], **self.tkwargs))
        ## optimize acqf
        candidates, acqf_values, sampled_subseq, sampled_id = cascade_optimize_acqf(
            acqf_dict=acqf_dict,
            bounds_list=bounds_list,
            q=self.q,
            num_restarts=10,
            raw_samples=512,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            subseq=subseq,
            costs=self.cost_ratios if self.cost_aware else None,
        )
        return candidates, acqf_values, sampled_subseq, sampled_id
    
    def msbo_step_fixed_xs(self, acqf_dict, subseq=None):
        ## create list of bounds for each subsequence of processes
        tunable_dims = []
        input_dims = []
        for _, summary in self.model.dataset.items():
            tunable_dims.append(summary['tunable'])
            input_dims.append(summary['x'].shape[-1])
        bounds_list = []
        for _ in range(len(tunable_dims)):
            n = sum(tunable_dims[1:]) + input_dims[0]
            bounds_list.append(torch.tensor([[0.0] * n, [1.0] * n], **self.tkwargs))
        ## optimize acqf
        candidates, acqf_values, sampled_subseq, sampled_id = cascade_optimize_acqf(
            acqf_dict=acqf_dict,
            bounds_list=bounds_list,
            q=self.q,
            num_restarts=10,
            raw_samples=512,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            subseq=subseq,
            costs=self.cost_ratios if self.cost_aware else None,
        )
        return candidates, acqf_values, sampled_subseq, sampled_id
    
    def optimize_acqf_and_get_observation(
            self, 
            acq_funcs: List[str], 
            sampler,
            model, 
            best_observed_value: float,
            beta: float,
            subseq: int = None,
    ):
        for acq_func in acq_funcs:
            if self.global_optimization:
                acqf = _init_acqf(
                    acq_func=acq_func, 
                    sampler=sampler,
                    model=model, 
                    best_observed_value=best_observed_value, 
                    beta=beta,
                )
                candidates, acqf_values, sampled_subseq, sampled_id = self.global_step(acqf=acqf)
                if len(acq_funcs)>1 and np.isclose(acqf_values, 0., atol=5.e-3):
                    continue
                else:
                    new_x = candidates.detach().cpu().numpy()
                    self.oracle.run_experiment(input={sampled_subseq: {'x': new_x}}, sample_id=sampled_id, global_input=True)
                    break
                # new_samples = self.oracle.get_data(sample_ids=[self.oracle.inventory.n_samples[-1]])
            else:
                acqf_dict = generate_acqf_dict(
                    acq_func=acq_func, 
                    sampler=sampler,
                    model=model, 
                    best_observed_value=best_observed_value,
                    duplicate_samples=self.duplicate_samples,
                    beta=beta,
                    fix_previous_xs=self.fixed_xs,
                )
                if self.fixed_xs:
                    candidates, acqf_values, sampled_subseq, sampled_id = self.msbo_step_fixed_xs(acqf_dict, subseq=subseq)    
                else:
                    candidates, acqf_values, sampled_subseq, sampled_id = self.msbo_step(acqf_dict, subseq=subseq)
                if len(acq_funcs)>1 and np.isclose(acqf_values, 0., atol=5.e-3):
                    continue
                else:
                    # extract controllable input to be passed to oracle
                    tunable_dim = self.model.dataset[sampled_subseq]['tunable']
                    new_x = candidates.detach().cpu().numpy()[..., :tunable_dim]
                    # run experiment with oracle
                    if sampled_subseq==0:
                        self.oracle.run_experiment(input={sampled_subseq: {'x': new_x}})
                        # new_samples = self.oracle.get_data(sample_ids=[self.oracle.inventory.n_samples[-1]])
                    else:
                        old_input = self.oracle.get_data(sample_ids=[sampled_id])
                        if max(old_input.keys())>=sampled_subseq:
                            input = {process_id: {'x': old_input[process_id]['x']} for process_id in range(sampled_subseq)}
                            input[sampled_subseq] = {'x': new_x}
                            self.oracle.run_experiment(input=input)
                            # new_samples = self.oracle.get_data(sample_ids=[self.oracle.inventory.n_samples[-1]])
                        else:
                            self.oracle.run_experiment(input={sampled_subseq: {'x': new_x}}, sample_id=sampled_id)
                            # new_samples = self.oracle.get_data(sample_ids=[sampled_id])
                    break
        # acq_type = 'EI' if 'qExpectedImprovement' in str(acq_func) else 'UCB'
        # print(f'Acquisition Function: {acq_type}')
        # print(f'Sampled subsequence: {sampled_subseq}')
        return sampled_subseq, sampled_id, new_x
    
    def random_sample(self):
        '''
        global random sampling
        '''
        n = sum([data['tunable'] for _, data in self.dataset.items()])
        sampled_subseq = 0    
        sampled_id = None
        new_x = np.random.uniform(size=(1, n))
        self.oracle.run_experiment(input={sampled_subseq: {'x': new_x}}, global_input=True)
        return sampled_subseq, sampled_id, new_x
    
    def _step_wrapper(self, step_func, condition):
        def wrapped_f(*args, **kwargs):
            if condition:
                self.global_optimization=True
                res = step_func(*args, **kwargs)
                self.global_optimization=False
            else:
                res = step_func(*args, **kwargs)
            return res
        return wrapped_f

    def subseq_selection(self, subseq_counter, subseq_reps):
        """
        minimial frequency
        """
        for previous_process in range(len(subseq_counter)-1):
            flag = self.duplicate_samples or self.oracle.inventory.available_samples(process_id=previous_process+1)>0
            counter = subseq_counter[previous_process]
            if flag and counter!=0 and counter%subseq_reps[previous_process]==0:
                subseq_counter[previous_process] = 0
                return previous_process+1
        return None

    def run(
            self,
            epochs: int = 10, 
            n_trials: int = 3, 
            mc_samples: int = 256, 
            initial_data: int = 8, 
            global_every: int = 5,  ### if I set this to 1 == astudillo paper
            save: bool = True,
            verbose: bool = True,
            init_method: str = 'sobol',
            random_search: bool = False,
            acq_funcs: List[str] = ['qEI'],
            cost_ratios: List[float] = [.5, .5],
            subseq_reps: Optional[List[int]] = None,
    ):
        self.cost_ratios = cost_ratios
        msbo_flag = not self.global_optimization #used when in msbo to set global iteration every 'global_every' epochs
        ## average over multiple trials
        for trial in range(1, n_trials + 1): 
            # if verbose:
            print(f"-Trial {trial:>2} of {n_trials} ", end="\n") 
            # initialize oracle, dataset and model   
            self.oracle.inventory.erase()
            self.generate_initial_data(n=initial_data, method=init_method)
            self.initialize_model() 
            
            ## run 'epochs' rounds of BayesOpt after the initial random batch
            # counter = 0 
            iteration = 1
            current_cost = 0.
            subseq_counter = [0]*len(self.oracle.process.n_processes)
            denoised_summary = {'x': [], 'y': []}
            max_mean_summary = {'x': [], 'y': []}
            while current_cost<epochs and iteration<500:
                ## fit the model
                self.model.fit()  
                if random_search:
                    sampled_subseq, sampled_id, new_x = self.random_sample()
                else:                    
                    # condition_glob = (iteration%global_every==0 or counter>=5) and msbo_flag
                    condition_glob = (
                        (
                            iteration%global_every==0 or current_cost+1.>=epochs
                        ) and msbo_flag
                    )
                    optimization_step = self._step_wrapper(self.optimize_acqf_and_get_observation, condition=condition_glob)
                    # subseq = self.subseq_selection(iteration, subseq_reps) if subseq_reps is not None else None # when forced frequency
                    subseq = self.subseq_selection(subseq_counter, subseq_reps) if subseq_reps is not None else None
                    sampled_subseq, sampled_id, new_x = optimization_step(
                        acq_funcs=acq_funcs, 
                        sampler=SobolQMCNormalSampler(mc_samples),
                        model=self.model, 
                        best_observed_value=self.best_observed_value,
                        beta=100*(epochs-current_cost)/epochs + 1,
                        subseq=subseq,
                    )
                    subseq_counter[sampled_subseq] += 1
                    if sampled_subseq>0:
                        subseq_counter[sampled_subseq-1] = 0

                # add new observations to dataset and update summary
                self._update_dataset()
                self.summary.update(
                    trial, 
                    sampled_subseq, 
                    sampled_id, 
                    new_x, 
                    self.best_observed_value, 
                    self.best_sample,
                )
                ## reinitialize the models so they are ready for fitting on next iteration
                ## use the current state dict to speed up fitting
                self.initialize_model(self.model.state_dict())

                if verbose:
                    print(
                        f"Batch {iteration:>2}: best_value = "
                        f"({self.best_observed_value:>4.2f}), "
                        f"best_sample = ({self.best_sample}), "
                        f"current cost = {current_cost}", end="\n"
                    ) 
                if save:
                    self.oracle.inventory_table.to_excel(self.save_path+self.name+'_trial'+str(trial)+'.xlsx')

                sample_counter = self.oracle.samples_per_process()
                current_cost = sum(np.asarray(sample_counter)*cost_ratios)
                iteration += 1
                
                ### COMPUTE DENOISED METRICS
                diff_process = DifferentiableProcess(self.oracle.process)

                ### COMPUTE DENOISED OBJECTIVE AT BEST OBSERVED POINT
                sample_dict = self.oracle.inventory.data_dict[self.best_sample]
                all_x = [sample_dict[pid]['x'].flatten() for pid in sorted(sample_dict.keys())]
                best_sample_x = np.concatenate(all_x)
                with torch.no_grad():
                    best_sample_y = diff_process(torch.from_numpy(best_sample_x)).item()
                denoised_summary['x'].append(best_sample_x)
                denoised_summary['y'].append(best_sample_y)
                print(f' -Denoised sample: {[f"{x:.3f}" for x in best_sample_x.tolist()]} -> {best_sample_y:.4f}')                
                
                ### Ground Truth of max mean across seen samples so far
                samples_ids = self.oracle.inventory.samples_at_process(process_id=1)
                data_dict = self.oracle.inventory.read(
                    sampleIds=samples_ids, 
                    processIds=self.oracle.process.n_processes, 
                    typeIds=['x', 'm'],
                )
                X = torch.stack([torch.tensor(np.concatenate([dd[0]['x'], dd[1]['x']])) for dd in data_dict.values()])
                self.model.eval()
                with torch.no_grad():
                    mean = self.model.forward(X).rsample(sample_shape=torch.Size([10])).mean(dim=0).squeeze()
                argmax_idx = torch.argmax(mean)
                best_x = X[argmax_idx]
                with torch.no_grad():
                    best_y = diff_process(best_x).item()
                max_mean_summary['x'].append(best_x.detach().cpu().numpy().squeeze())
                max_mean_summary['y'].append(best_y)
                print(f' -Max mean {[f"{x:.3f}" for x in best_x.tolist()]} -> {best_y:.4f}')   

        return self.summary, denoised_summary, max_mean_summary
