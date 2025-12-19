import os
import yaml
import argparse
import numpy as np
from msbo.simulator.laboratory import Laboratory
from msbo.optimizer.BO import BayesOptimizer
from msbo.optimizer.utils import OptimizationSummary



parser = argparse.ArgumentParser()
parser.add_argument('-p', '--process', type=str, default='standard')
parser.add_argument('-c', '--complexity', nargs="+", type=int, default=[50, 2])
parser.add_argument('-sr', '--subseq_reps', nargs="+", type=int, default=None)
parser.add_argument('-sc', '--start_counter', type=int, default=0)
args = parser.parse_args()


N_EXPERIMENTS = 1
PATH = './results/'+args.process+'_'+'_'.join([str(c) for c in args.complexity])+'/'
if not os.path.exists(PATH):
    os.mkdir(PATH)
CONFIG = args.process+'.yml'

seed = args.start_counter
counter = 0
while counter<N_EXPERIMENTS:
    try:
        # Create the directory
        SAVE_PATH = PATH+str(seed)+'/'
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)

        with open('./config/'+CONFIG, 'r') as f:
            platform_config = yaml.full_load(f)
        platform_config['complexity'] = {i:c for i, c in enumerate(args.complexity)}
        platform_config['seed'] = seed    
        platform = Laboratory(platform_config)
        x_optimum, optimum = platform.get_maximum()
        if optimum < 0.:
            raise ValueError(f"Negative optimum can lead to error when choosing the acquisition function value (Not handled correctly).")
        print(f'seed: {seed} - optimum: {optimum}')
        with open(os.path.join(SAVE_PATH, f"ground_truth.txt"), "w") as f:
            f.write(f"{x_optimum.tolist()}, {optimum}\n")
        initial_data = 2*(sum([v for v in platform_config['input_dim'].values()]) + 1)

        N_TRIALS = 1
        ITER = 100 + initial_data  # sets the maximum budget (cost)
        STEPS = 2
        optim_config = [
            {'name': 'msbo_EI_UCB', 'epochs': ITER, 'global_every': 1000, 'global_optimization': False, 'acqf': ['qEI', 'qUCB']},
            {'name': 'random', 'epochs': ITER, 'global_every': 1000, 'global_optimization': True, 'random_search':True, 'acqf': None},
            {'name': 'global_EI', 'epochs': ITER, 'global_every': 1000, 'global_optimization': True, 'acqf': ['qEI']},
            # {'name': 'astudillo_EI', 'epochs': ITER, 'global_every': 1, 'global_optimization': False, 'acqf': ['qEI']},
        ]

        for config in optim_config:
            print(f"Optimizer: {config['name']}")
            optimizer = BayesOptimizer(
                oracle=platform, 
                save_path=SAVE_PATH, 
                name=config['name'], 
                global_optimization=config['global_optimization'],
                duplicate_samples=False,
                cost_aware=False,
                fixed_xs=config.get('fixed_xs', False),
            )
            summary, denoised_summary, max_mean_summary = optimizer.run(
                epochs=config['epochs'], 
                n_trials=N_TRIALS, 
                initial_data=initial_data, 
                verbose=True, 
                save=True, 
                global_every=config['global_every'],
                random_search=config.get('random_search', False),
                acq_funcs=config['acqf'],
                cost_ratios=[.02, .98], 
                subseq_reps=args.subseq_reps,
            )
            np.savez(
                os.path.join(SAVE_PATH, f"{config['name']}_denoised.npz"),
                x=np.array([x for x in denoised_summary['x']]),
                y=np.array(denoised_summary['y'])
            )
            np.savez(
                os.path.join(SAVE_PATH, f"{config['name']}_max_mean.npz"),
                x=np.array([x for x in max_mean_summary['x']]),
                y=np.array(max_mean_summary['y'])
            )
            OptimizationSummary.compute_regrets(summary, optimum=optimum)
            summary.save_json(name=config['name'], path=SAVE_PATH)

            OptimizationSummary.plot(
                title_prefix=config['name'], 
                save=True, 
                path=SAVE_PATH, 
                **{
                    config['name']:summary.datadict, 
                    'steps':STEPS,
                    'best_value': optimum,
                },
            )
            OptimizationSummary.plot(
                title_prefix=config['name'], 
                save=True, 
                path=SAVE_PATH, 
                regret=True,
                **{
                    config['name']:summary.datadict, 
                    'steps':STEPS,
                },
            )
            
        counter += 1
        seed += 1
        
    except Exception as e:
        print(f'Exception {e} with seed {seed}')
        seed += 1


