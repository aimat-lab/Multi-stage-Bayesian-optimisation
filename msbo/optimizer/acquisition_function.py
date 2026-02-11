import torch
from typing import List, Optional

from botorch.optim import optimize_acqf
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound

from msbo.optimizer.utils import ids_up_to


def _init_acqf(
        acq_func: str,
        sampler,
        model: ModelListGP, 
        best_observed_value: float, 
        beta: float,
    ):
    if acq_func=='qEI':
        acqf = qExpectedImprovement(
            model=model,
            best_f=best_observed_value,
            sampler=sampler,
        )
    elif acq_func=='qUCB':
        acqf = qUpperConfidenceBound(
            model=model,
            beta=beta,
            sampler=sampler,
        )
    return acqf


def generate_acqf_dict(
        acq_func, 
        sampler,
        model: ModelListGP, 
        best_observed_value: float, 
        beta: float,
        duplicate_samples: bool = False,
        fix_previous_xs: bool = False, 
) -> List:
    acqf_dict = dict()
    acqf_dict[0] = {
        'acqf_list': [_init_acqf(
            acq_func,
            sampler,
            model, 
            best_observed_value, 
            beta,
            )],
        'fixed_vals_list': [],  
        'sample_id': None,
    }
    for idx in range(1, len(model.GPs)):
        available_samples_id = ids_up_to(model.dataset, idx-1)    
        acqf_dict[idx] = dict()
        acqf_dict[idx]['acqf_list'] = []
        acqf_dict[idx]['fixed_vals_list'] = []
        acqf_dict[idx]['sample_id'] = []
        if fix_previous_xs:
            # Fix all previous xs for each sample
            for sample_idx, sample_id in enumerate(model.dataset[idx-1]['ids']):
                if not duplicate_samples and not sample_id.item() in available_samples_id:
                    continue
                prev_xs = []
                for j in range(idx):
                    ids_j = model.dataset[j]['ids']
                    try:
                        idx_in_j = (ids_j == sample_id).nonzero(as_tuple=True)[0].item()
                    except Exception:
                        continue
                    x_j = model.dataset[j]['x'][idx_in_j]
                    prev_xs.append(x_j)
                if prev_xs:
                    fixed_vals = torch.cat(prev_xs, dim=-1)
                    fixed_cols = torch.arange(0, fixed_vals.shape[-1])
                    acqf_dict[idx]['acqf_list'].append(_init_acqf(
                        acq_func,
                        sampler,
                        model,  # use full chain
                        best_observed_value, 
                        beta,
                    ))
                    acqf_dict[idx]['fixed_vals_list'].append({col.item(): val.item() for col, val in zip(fixed_cols, fixed_vals)})
                    acqf_dict[idx]['sample_id'].append(sample_id)
        else:
            # Original behavior: fix y and use subsequence
            fixed_vals = model.dataset[idx-1]['y']
            if model.dataset[idx]['residual']:
                dim_x_prev = model.dataset[idx-1]['tunable']
                x_prev = model.dataset[idx-1]['x'][..., :dim_x_prev]
                fixed_vals = torch.cat([x_prev, fixed_vals], dim=-1)
            partial_model = model.extract_subsequence(idx=idx) 
            first_tunable_dim = partial_model.dataset[0]['tunable']
            fixed_cols = torch.arange(first_tunable_dim, first_tunable_dim+fixed_vals.shape[-1])
            for f_vals, sample_id in zip(fixed_vals, model.dataset[idx-1]['ids']):
                if not duplicate_samples and not sample_id.item() in available_samples_id:
                    continue
                acqf_dict[idx]['acqf_list'].append(_init_acqf(
                        acq_func,
                        sampler,
                        partial_model, 
                        best_observed_value, 
                        beta,
                    )
                )
                acqf_dict[idx]['fixed_vals_list'].append({col.item(): val.item() for col, val in zip(fixed_cols, f_vals)})
                acqf_dict[idx]['sample_id'].append(sample_id)
    return acqf_dict


def cascade_optimize_acqf(
        acqf_dict,
        bounds_list,
        q,
        num_restarts,
        raw_samples, 
        options,
        subseq = None, 
        costs: Optional[List[float]] = None,
        ):
    '''
    works only for q=1 at the moment
    ''' 
    if costs is not None:
        if any(abs(c) < 1e-6 for c in costs):
            raise ValueError("Costs must be greater than 1e-6 for numerical stability.")
    candidates_list = []
    acqf_values_list = []
    sampled_id_list = []
    for process_id, process_dict in acqf_dict.items():
        if subseq is not None and process_id!=subseq:
            print(f'skipping process_id: {process_id} because subseq: {subseq}')
            candidates=[None]
            acqf_values=-torch.inf
            sample_id=None
        else:
            if len(process_dict['acqf_list'])==0:
                candidates=[None]
                acqf_values=-torch.inf
                sample_id=None
            else:
                if len(process_dict['fixed_vals_list'])==0:
                    candidates, acqf_values = optimize_acqf(
                        acq_function=process_dict['acqf_list'][0],
                        bounds=bounds_list[process_id],
                        q=q,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,  
                        options=options,
                    )    
                    sample_id = None         
                else:
                    temp_candidates = []
                    temp_acqf_vals = []
                    temp_sample_ids = []
                    for acqf, fixed_vals, sample_id in zip(process_dict['acqf_list'], process_dict['fixed_vals_list'], process_dict['sample_id']):
                        candidates, acqf_values = optimize_acqf(
                            acq_function=acqf,
                            bounds=bounds_list[process_id],
                            q=q,
                            fixed_features=fixed_vals,
                            num_restarts=num_restarts,
                            raw_samples=raw_samples,  
                            options=options,
                        )
                        temp_candidates.append(candidates)
                        temp_acqf_vals.append(acqf_values)  
                        temp_sample_ids.append(int(sample_id))
                    selected = torch.argmax(torch.tensor(temp_acqf_vals))
                    candidates = temp_candidates[selected]
                    acqf_values = temp_acqf_vals[selected]
                    sample_id = temp_sample_ids[selected]
        candidates_list.append(candidates)
        acqf_values_list.append(acqf_values)
        sampled_id_list.append(sample_id)
    if costs is not None:
        acqf_values_list = [v/c for v, c in zip(acqf_values_list, costs)]
    sampled_subseq = torch.argmax(torch.tensor(acqf_values_list)).item()
    return (
        candidates_list[sampled_subseq], 
        acqf_values_list[sampled_subseq],
        sampled_subseq,
        sampled_id_list[sampled_subseq],
    )

