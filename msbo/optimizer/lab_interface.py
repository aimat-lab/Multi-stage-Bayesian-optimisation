import torch
import numpy as np

from msbo.optimizer.utils import at_least_2dim


def get_dataset_from_platform(platform, tkwargs={"dtype": torch.double, "device": torch.device("cpu")}, msopt=True):
    model_dataset = dict()
    if platform.inventory.n_samples is None:
        return model_dataset
    if msopt:
        for step in platform.process.n_processes:
            model_dataset[step] = dict()
            sample_ids = platform.inventory.samples_at_process(step)    # all samples that arrive to process "step"
            data = platform.get_data(sample_ids=sample_ids, process_ids=[step])[step]
            # concatenate previous output (and when filter also previous input): x <- x_t + x_t-1 + m_t-1
            if platform.config['measurement'].get(step-1, False):
                m = platform.get_data(sample_ids=sample_ids, process_ids=[step-1])[step-1]['m']
                if platform.config['measurement'][step-1][0] == 'filter':
                    x = platform.get_data(sample_ids=sample_ids, process_ids=[step-1])[step-1]['x']
                    m = np.concatenate([at_least_2dim(x), at_least_2dim(m)], axis=-1)
                x = np.concatenate([at_least_2dim(data['x']), at_least_2dim(m)], axis=-1)
            else:
                x = data['x']
            model_dataset[step] = {
                'x': torch.tensor(x, **tkwargs), 
                'y': torch.tensor(data['m'], **tkwargs), 
                'ids': torch.tensor(sample_ids, **tkwargs),
                'tunable': platform.config['input_dim'][step], 
                'residual': platform.config['measurement'].get(step-1, [False])[0]=='filter',
            }
    else:
        model_dataset = dict()
        last_step = platform.process.n_processes[-1]
        sample_ids = platform.inventory.samples_at_process(last_step)
        data = platform.get_data(sample_ids=sample_ids)
        x = []
        for step in platform.process.n_processes:
            x.append(at_least_2dim(data[step]['x']))
        x = np.concatenate(x, axis=-1)
        model_dataset[0] = {
            'x': torch.tensor(x, **tkwargs), 
            'y': torch.tensor(data[last_step]['m'], **tkwargs), 
            'ids': torch.tensor(sample_ids, **tkwargs),
            'tunable': x.shape[-1],
        }
    return model_dataset
            

