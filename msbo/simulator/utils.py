import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from collections.abc import Callable
from typing import Optional, List, Dict


def timit(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        print(f'took {time.time() - start} sec')
        return out
    return wrapper

    
def at_least_2dim(x: np.ndarray) -> np.ndarray:
    if len(x.shape)<2:
        x = x.reshape(-1, 1)
    return x     


def plot_2d(data_gener, limits=(0,1), name='', save=True):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    if isinstance(data_gener, dict):
        z = at_least_2dim(data_gener['objective'])
        x = at_least_2dim(data_gener['parameters'][:, 0])
        y = at_least_2dim(data_gener['parameters'][:, 1])
        scatter = ax.scatter(x, y, z, cmap=cm.coolwarm, alpha=.6, s=10, c=z)
        fig.colorbar(scatter, shrink=0.5, aspect=5)
    else:
        a = np.linspace(limits[0], limits[1], 100)
        y, x = np.meshgrid(a,a)
        X = np.stack((x,y), axis=2).reshape(-1,2)
        z = data_gener(X)
        if isinstance(z, dict):
            z = z['objective']
        elif isinstance(z, tuple):
            z = z[0]
        z = z.reshape(x.shape)
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.3)
        fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=50., azim=-103.)
    plt.title(name)
    if save:
        plt.savefig(name+'.png')
    else:
        plt.show()
    plt.close()
#    plt.hist(z.flatten(), bins=10)
#    plt.show()
#    plt.close()
    
    
def plot_1d(data_gener, limits=(0,1), name='', save=True):
    fig, ax = plt.subplots()
    if isinstance(data_gener, dict):
        x = data_gener['parameters']
        z = data_gener['objective']
        ax.scatter(x, z, color='red')    
    else:
        x = np.linspace(limits[0], limits[1], 100).reshape(-1,1)
        z = data_gener(x)
        ax.plot(x, z, color='red')
        
    ax.grid()
    plt.title(name)
    if save:
        plt.savefig(name+'.png')
    else:
        plt.show()
    plt.close()   


def split_dataset(n_samples: int, test_ratio: float = .33, rng=None):
    '''
    given the length of the dataset (n_samples) it returns two list of indices randomly split for train and test
    '''
    if rng is None:
        rng = np.random.default_rng()
    n_test = int(n_samples*test_ratio)
    idx = rng.permutation(n_samples)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


############################  Dict conversions  ##############################################################

def dataDict_to_tableDict(dataDict: Dict) -> Dict:
    '''
    dataDict = {sample_id: {process_id:{'x': ndarray, 'h': ndarray, 'm': ndarray}, etc...}, etc...}
    tableDict = {sample_id: {(process_id, 'x'): ndarray, (process_id, 'h'): ndarray, (process_id, 'm'): ndarray, etc...}, etc...}
    '''
    return {
        sampleId: {
            (processId, typeId): values
            for processId, processDict in sampleDict.items()
            for typeId, values in processDict.items()
        }
        for sampleId, sampleDict in dataDict.items()
    }


def vecDict_to_dataDict(vecDict: Dict, sample_ids_list: Optional[List[int]] = None) -> Dict:
    '''
    vecDict = {process_id: {'x': ndarray of shape (n_samples, dim x), 'h': same, 'm': same}, etc...}
    dataDict = {sample_id: {process_id:{'x': ndarray, 'h': ndarray, 'm': ndarray}, etc...}, etc...}
    '''
    if sample_ids_list is None:
        n_samples = list(vecDict.values())[0]['x'].shape[0]
        sample_ids_list = list(range(n_samples))
    dataDict = {
        sampleId: {
            processId: {
                typeId: values[sample_idx, :] for typeId, values in processDict.items() if isinstance(values, np.ndarray)
            } for processId, processDict in vecDict.items()
        } for sample_idx, sampleId in enumerate(sample_ids_list)
    }
    return dataDict


def dataDict_to_vecDict(dataDict: Dict) -> Dict:
    '''
    dataDict = {sample_id: {process_id:{'x': ndarray, 'h': ndarray, 'm': ndarray}, etc...}, etc...}
    vecDict = {process_id: {'x': ndarray of shape (n_samples, dim x), 'h': same, 'm': same}, etc...}

    CAREFUL! this function assumes that there are not missing values, i.e. all samples in dataDict have values for all processes
    '''
    n_samples = len(dataDict)
    vecDict = {
        processId: {
            typeId: np.empty((n_samples, values.shape[-1]))*np.nan  #nans added to be sure we don't forget that we assume all samples in dataDict to have all values
            for typeId, values in processDict.items()
        } for _, sampleDict in dataDict.items() 
          for processId, processDict in sampleDict.items()
    }
    for idx, (_, sampleDict) in enumerate(dataDict.items()):
        for processId, processDict in sampleDict.items():
            for typeId, values in processDict.items():
                vecDict[processId][typeId][idx, :] = values
    return vecDict














