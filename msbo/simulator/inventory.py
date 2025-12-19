from typing import List, Dict, Optional


class Inventory:
    def __init__(
            self,
            data_dict: Dict = dict(),
    ) -> None:
        '''
        dataDict = {sample_id: {process_id:{'x': ndarray, 'h': ndarray, 'm': ndarray}, etc...}, etc...}
        '''
        super().__init__()
        self.data_dict = data_dict

    @property
    def n_samples(self) -> List[int]:
        if self.data_dict:
            return list(self.data_dict.keys())
        else:
            return None
    
    def read(
            self,
            sampleIds: List[int],
            processIds: List[int],
            typeIds: List[str],
    ) -> Dict:
        return {
            sample_id: {
                process_id : {
                    type_id: self.data_dict[sample_id][process_id][type_id] 
                    for type_id in typeIds if type_id in self.data_dict[sample_id][process_id]
                } for process_id in processIds if process_id in self.data_dict[sample_id]
            } for sample_id in sampleIds if sample_id in self.data_dict
        }

    def write(
            self, 
            data: Dict,
    ) -> None:
        for sample_id, sampleDict in data.items():
            if not sample_id in self.data_dict:
                self.data_dict[sample_id] = dict()
            for process_id, processDict in sampleDict.items():
                if not process_id in self.data_dict[sample_id]:
                    self.data_dict[sample_id][process_id] = dict()
                for type_id, values in processDict.items():
                    self.data_dict[sample_id][process_id][type_id] = values       

    def samples_at_process(self, process_id: int) -> List[int]:
        '''
        return list of ids corresponding to samples that have been processed ONLY UP TO process process_id
        '''
        return [sample_id for sample_id, sampleDict in self.data_dict.items() if process_id in sampleDict]
    
    def processes_at_sample(self, sample_id: int) -> List[int]:
        '''
        return list of ids corresponding to steps that have been processed for sample "sample_id"
        '''
        sampleDict = self.data_dict.get(sample_id, None)
        if sampleDict is None:
            process_ids = []
        else:
            process_ids = list(sampleDict.keys())
        return process_ids
    
    def count_samples(self, n_processes):
        counter = []
        for i in range(n_processes):
            counter.append(len(self.samples_at_process(process_id=i)))
        return counter
    
    def erase(self):
        self.data_dict = dict()
    
    def available_samples(self, process_id):
        assert process_id>0, 'process_id has to be > 0'
        prev_process_samples = len(self.samples_at_process(process_id=process_id-1))
        actual_process_samples = len(self.samples_at_process(process_id=process_id))
        return prev_process_samples-actual_process_samples

