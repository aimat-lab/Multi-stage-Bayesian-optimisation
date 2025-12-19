import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from typing import List, Optional, Tuple


def timit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        print(f'took {time.time() - start} sec')
        return out
    return wrapper
    
    
def at_least_2dim(x):
    if len(x.shape)<2:
        x = x.reshape(-1, 1)
    return x


def ids_up_to(dataset, idx):
    current_idx = set([v.item() for v in dataset[idx]['ids']])
    next_idx = dataset.get(idx+1, None)
    if next_idx is None:
        next_idx = set()
    else:
        next_idx = set([v.item() for v in next_idx['ids']])
    return current_idx-next_idx         


def get_stats(
        name,
        n_trials: int = 5,
    ):
    summary = []
    for trial in range(1, n_trials+1):
        df = pd.read_excel(
            name+'_trial'+str(trial)+'.xlsx',
            header=[0, 1], 
            index_col=[0],
        )
        df = df.where(pd.notnull(df), '[nan]')
        last_process = max(set([process_id for process_id, _ in df.columns]))
        res = np.array([float(val[1:-1]) if val[1:-1]!='nan' else -np.inf for val in df.loc[:, (last_process, 'm')]])
        summary.append(np.array([max(res[:i])for i in range(1, len(res))], dtype=float))
    max_length = 0
    for arr in summary:
        max_length = max(max_length, len(arr))
    for i in range(n_trials):
        if len(summary[i])<max_length:
            arr = np.ones(max_length)*max(summary[i])
            arr[:len(summary[i])] = summary[i]
            summary[i] = arr
    summary = np.stack(summary, axis=0)
    return summary.mean(axis=0), summary.std(axis=0)


def plot_stats(
        names_list,
        n_trials_list, 
        path,
        x_init: int = 22,
        title: str = 'Optimization summary',
        xlabel: str = '#samples',
        ylabel: str = 'objective',
        colors: List[str] = ['blue', 'green', 'red', 'brown', 'purple'],
        save: bool = True,
    ):    
    if not isinstance(n_trials_list, list):
        n_trials_list = [n_trials_list]*len(names_list)

    summary = dict()
    for name, n_trials in zip(names_list, n_trials_list):
        summary[name] = dict()
        mean, std = get_stats(path+name, n_trials)
        summary[name.split('/')[-1]]['mean'] = mean
        summary[name.split('/')[-1]]['std'] = std
    
    plt.rcParams["figure.figsize"] = (12,8)
    for c, (name, stats) in enumerate(summary.items()):
        xx = np.arange(1, stats['mean'].shape[0]+1)
        plt.plot(xx, stats['mean'], color=colors[c], label=name)
        plt.fill_between(
            xx, 
            stats['mean']+2*stats['std'], 
            stats['mean']-2*stats['std'], 
            alpha=0.1, 
            edgecolor='tab:'+colors[c], 
            color=colors[c]
        )

    plt.vlines(x_init, .0, 1.9, colors='black', linestyles='dashed', label='rnd_init')
    # plt.vlines(x_init, min(mean)-max(std), max(mean)+max(std), colors='red', linestyles='dashed', label='rnd_init')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(.0, 1.9)
    plt.xlim(0, 60)
    plt.legend()
    plt.grid()
    plt.title(title)

    if save:
        plt.savefig(path+title+'.png')
    else:
        plt.show()
    plt.close()


def count_samples(
        figname,
        path: str = './',
        save: bool = False,
    ):
    colors = ['blue', 'cyan', 'olive', 'gray', 'green', 'red', 'orange', 'purple', 'pink', 'brown', 'black']
    summary = dict()
    for subdir, _, files in os.walk(path):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".xlsx"):
                name = file.split('_trial')[0]
                if name not in summary:
                    summary[name] = dict()
                df = pd.read_excel(
                    filepath,
                    header=[0, 1], 
                    index_col=[0],
                )
                processes = set([process_id for process_id, _ in df.columns])
                for k in processes:
                    if summary[name].get(k, False):
                        summary[name][k].append(len(df.index) - sum(df.loc[:, (k, 'm')].isna()))
                    else:
                        # print(f'name: {name}, k: {k}')
                        # print(df)
                        summary[name][k] = [len(df.index) - sum(df.loc[:, (k, 'm')].isna())]
    
    x = np.arange(len(summary[name]))
    width = .2
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained', figsize=(10,8))
    for c, name in enumerate(summary.keys()):
        if 'msbo' in name or name=='global_EI':
            means, stds = [], []
            for k in summary[name].keys():
                means.append(np.mean(summary[name][k]))
                stds.append(np.std(summary[name][k])/np.sqrt(len(summary[name][k])))
            offset = width * multiplier
            rects = ax.bar(x + offset, means, yerr=stds, width=width , alpha=.5, color=colors[c], edgecolor='black', label=name)
            ax.bar_label(rects, padding=3, fmt='%.2f')
            multiplier += 1
    ax.set_xlabel('subprocess id')
    ax.set_ylabel('#samples')
    ax.set_title('Subprocesses sampling ratio')
    ax.set_xticks(x + width, [str(k) for k in summary[name].keys()])
    ax.legend(loc='upper right', ncols=2)
    if save:
        plt.savefig(path+figname+'_hist.png')
    else:
        plt.show()
    plt.close()
    

def regret_summary(path: str, steps: int, costs: List[float], save: bool = True, title: str = ''):
    xlabel = '#iterations'
    ylabel = 'regret'
    colors = ['blue', 'cyan', 'green', 'olive', 'red', 'orange', 'purple', 'pink', 'brown', 'gray', 'black']

    stats = dict()
    cost_stats = dict()
    for subdir, _, files in os.walk(path):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".json"):
                name = file.removesuffix('_summary.json')
                # if name not in ['random', 'global_UCB', 'astudillo_UCB', 'msbo_EI_UCB', 'msbo_UCB']:
                #     continue
                if name not in stats:
                    stats[name] = []
                    if 'msbo' in name:
                        cost_stats[name] = []
                with open(filepath, 'r') as f:
                    summary = json.load(f)
                for _, trial_summary in summary.items():
                    regret = [iter['best_observed_value'] for iter in trial_summary]
                    # regret = [iter['regret'] for iter in trial_summary]
                    stats[name].append(regret)
                    if 'msbo' in name:
                        cost = [costs[iter['sampled_subseq']] for iter in trial_summary]
                        cost_stats[name].append(cost)
    
    plt.rcParams["figure.figsize"] = (12,8)
    for c, name in enumerate(stats.keys()):
        if name.startswith('msbo'):
            max_length = max(len(arr) for arr in cost_stats[name])
            padded_costs = [np.pad(arr, (0, max_length - len(arr)), 'constant') for arr in cost_stats[name]]
            padded_stats = [np.pad(arr, (0, max_length - len(arr)), 'edge') for arr in stats[name]]
            cost_stats[name] = np.vstack(padded_costs)
            stats[name] = np.vstack(padded_stats)
        mean = np.asarray(stats[name]).mean(axis=0)
        std = np.asarray(stats[name]).std(axis=0)/np.sqrt(len(stats[name]))

        xx = np.arange(1, mean.shape[0]+1)
        if name.startswith('msbo'):
            xx = np.asarray(cost_stats[name]).mean(axis=0)
            xx = np.asarray([xx[:i+1].sum() for i in range(len(xx))])

        plt.plot(xx, mean, color=colors[c], label=name)
        plt.fill_between(
            xx, 
            mean+1*std, 
            mean-1*std, 
            alpha=0.1, 
            edgecolor='tab:'+colors[c], 
            color=colors[c]
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower left')
    plt.grid()
    plt.title('Methods Summary')
    if save:
        plt.savefig(path+'Methods Summary '+title+'.png')
    else:
        plt.show()
    plt.close()


class OptimizationSummary:
    def __init__(self) -> None:
        self.datadict = dict()

    def init_trial(self) -> None:
        pass

    def update(
            self, 
            trial, 
            sampled_subseq, 
            sampled_id, 
            new_x, 
            best_observed_value, 
            best_sample,
    ):
        if not self.datadict.get(trial, False):
            self.datadict[trial] = []
        self.datadict[trial].append(
            {
                'sampled_subseq': sampled_subseq, 
                'sampled_id': sampled_id, 
                'new_x': new_x.tolist(),
                'best_observed_value': best_observed_value,
                'best_sample': best_sample,
            }
        )

    @staticmethod
    def compute_regrets(datadict, optimum: float):
        '''log_10(|best - current_best|)'''
        if not isinstance(datadict, dict):
            datadict = datadict.datadict
        for _, trial_summary in datadict.items():
            for _, step_summary in enumerate(trial_summary):
                current_best = step_summary['best_observed_value']
                opt_dist = max((optimum - current_best).item(), 0)
                regret = np.log10(opt_dist+1e-10).item()
                step_summary['regret'] = regret 
                step_summary['opt_dist'] = opt_dist
    
    def save_json(self, name, path: Optional[str] = './'):
        with open(path+name+'_summary.json', 'w') as f:
            json.dump(self.datadict, f)    
    
    @staticmethod
    def plot(
        title_prefix: str = '', 
        save: bool = True, 
        path: Optional[str] = './', 
        regret: bool = False,
        **kwargs,
    ):
        title = 'Optimization summary'
        xlabel = '#iterations'
        if regret:
            ylabel = 'regret'
        else:
            ylabel = 'best objective value'
        colors = ['blue', 'cyan', 'green', 'olive', 'red', 'orange', 'purple', 'pink', 'brown', 'gray', 'black']
        
        stats = dict()
        for name, summary in kwargs.items():
            if name=='steps' or name=='best_value':
                continue
            stats[name] = dict()
            best_value = []  # might be regret, too laxy to change name
            for trial, data in summary.items():
                if regret:
                    best_value.append([iter['regret'] for iter in data])
                else:
                    best_value.append([iter['best_observed_value'] for iter in data])
            if int(trial)>1:
                best_value = np.stack(best_value, axis=0)
                stats[name]['mean'] = best_value.mean(axis=0)
                stats[name]['std'] = best_value.std(axis=0)/np.sqrt(best_value.shape[0])
            else:
                stats[name]['mean'] = np.array(best_value).squeeze()
                stats[name]['std'] = np.zeros_like(stats[name]['mean'])
        
        plt.rcParams["figure.figsize"] = (12,8)
        for c, (name, values) in enumerate(stats.items()):
            xx = np.arange(1, values['mean'].shape[0]+1)
            if not name.startswith('msbo'):
                xx = kwargs['steps']*xx
            plt.plot(xx, values['mean'], color=colors[c], label=name)
            plt.fill_between(
                xx, 
                values['mean']+2*values['std'], 
                values['mean']-2*values['std'], 
                alpha=0.1, 
                edgecolor='tab:'+colors[c], 
                color=colors[c]
            )
        if (not regret) and kwargs.get('best_value', False):
            plt.hlines(y=[kwargs['best_value']], xmin=0, xmax=xx[-1], colors='black', linestyles='dashed', label='optimum')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if regret:
            plt.legend(loc='lower left')
        else:
            plt.legend(loc='lower right')
        plt.grid()
        plt.title(title)
        if save:
            if regret:
                title = title + ' regret'
            plt.savefig(path+title_prefix+'_'+title+'.png')
        else:
            plt.show()
        plt.close()
