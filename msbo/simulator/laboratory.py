import warnings
import pandas as pd

from typing import Optional, Dict, Union, List

from msbo.simulator.utils import dataDict_to_tableDict, dataDict_to_vecDict, vecDict_to_dataDict
from msbo.simulator.process import MultiStepProcess
from msbo.simulator.inventory import Inventory
from msbo.simulator.grad_optim import get_maximum


class Laboratory:
    def __init__(
            self, 
            config: Dict,
    ) -> None:
        self.config = config
        self.seed = config.get('seed', None)
        if self.seed is None: 
            warnings.warn("if seed is not passed to laboratory, results may be unreproducible")
        self.process = MultiStepProcess(config, seed=self.seed, last_squashed=False)
        self.inventory = Inventory()

    @property        
    def inventory_table(self) -> Union[pd.DataFrame, None]:
        '''
        returns a pandas DataFrame of the entire inventory
        '''
        if not self.inventory.data_dict:
            return None
        table_dict = dataDict_to_tableDict(self.inventory.data_dict)
        df = pd.DataFrame.from_dict(table_dict, orient='index')
        df = df.rename_axis(['sample ID'])
        df.columns = df.columns.set_names(['process ID', None])
        return df
    
    def samples_per_process(self) -> List[int]:
        n_processes = len(self.process.n_processes)
        counter = self.inventory.count_samples(n_processes=n_processes)
        return counter
    
    def randomized_sampling(
            self,
            n_samples: int,
            method: str = 'sobol',
    ) -> None:
        samples_vecDict = self.process.sample(
            n_samples=n_samples, 
            method=method,
        )
        self.inventory.write(
            vecDict_to_dataDict(samples_vecDict)
        )
    
    def store_data(
            self, 
            vecDict: Dict, 
            sample_ids_list: Optional[List[int]] = None,
    ) -> None:
        if sample_ids_list is None:
            old_samples_n = self.inventory.n_samples
            new_samples_n = list(vecDict.values())[0]['h'].shape[0]
            sample_ids_list = list(range(old_samples_n, new_samples_n+old_samples_n))
        dataDict = vecDict_to_dataDict(vecDict=vecDict, sample_ids_list=sample_ids_list)
        self.inventory.write(dataDict)
    
    def get_data(
            self, 
            sample_ids: Optional[List[int]] = None,
            process_ids: Optional[List[int]] = None,
    ) -> Union[Dict, List[Dict]]:
        '''
        returns data in a vecDict format
        if process_id (sample_id) is not specified it returns all processes (samples)
        '''
        if process_ids is None:
            process_ids = self.process.n_processes
        if sample_ids is None:
            sample_ids = self.inventory.n_samples

        data_dict = self.inventory.read(
            sampleIds=sample_ids, 
            processIds=process_ids, 
            typeIds=['x', 'm'],
        )
        return dataDict_to_vecDict(data_dict)
    
    def get_maximum(
            self,
            raw_samples: int = 512,
            batch_limit: int = 16,
    ):
        x_max, y_max = get_maximum(
            process=self.process,
            seed=self.seed,
            raw_samples=raw_samples,
            batch_limit=batch_limit,
        )
        return x_max, y_max

    def run_experiment(
            self, 
            input: Dict = {0: {'x': None}}, #vecDict
            sample_id: Optional[int] = None,
            global_input: bool = False,
    ):
        '''
        at the moment it can only manage sequential experiments, one sample from one process at a time
        '''
        if global_input:
            X = input[0]['x']
            input = dict()
            current_col = 0
            for process_id, dim in self.config['input_dim'].items():
                input[process_id] = dict()
                input[process_id]['x'] = X[..., current_col:current_col+dim]
                current_col += dim
                
        if sample_id is not None:
            last_process_id = self.inventory.processes_at_sample(sample_id)[-1]
            data_dict = self.inventory.read(
                sampleIds=[sample_id], 
                processIds=[last_process_id], 
                typeIds=['h'],
            )
            vecDict = dataDict_to_vecDict(data_dict)
        else:
            sample_id = self.inventory.n_samples[-1] + 1
            vecDict = dict()            

        for k, v in input.items():
            vecDict[k] = {'x': None}
            vecDict[k]['x'] = v['x']
        h_list, m_list = self.process(data=vecDict)

        for i, k in enumerate(input.keys()):
            vecDict[k]['h'] = h_list[i]
            vecDict[k]['m'] = m_list[i]
        self.store_data(vecDict=vecDict, sample_ids_list=[sample_id])
    