# Script to monitor the SMAC runs and kill and restart if they exceed the memory allowed

import os
import json
import time
import signal
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Monitor SMAC runs and kill if they exceed the number of consecutive config runs')
parser.add_argument('--run_name', type=str, default='None',
                    help='Name of the run to monitor')
parser.add_argument('--model', type=str, default='FTTransformer',
                    choices=['FTTransformer', 'ResNet', 'MLP',
                             'ResNetPre', 'FTTransformerPre',
                             'ResNetPost', 'FTTransformerPost'],
                    help="which model to use for NAS")
parser.add_argument('--prepost', type=str, default="None",
                    help="Pre/Post-processing technique to use if such a `model` is chosen")
parser.add_argument('--weighting', type=int, default=1,
                    help="amount of weighting to apply to fairness metric: 1 is no-weighting -> uses ParEgo")
parser.add_argument('--dataset', type=str, default='adult',
                    help='Dataset to run method')
parser.add_argument('--privilege_mode', type=str, default='sex',
                    help='Privilege mode to run method')
parser.add_argument('--num_configs_switch', type=int, default=50,
                    help='Number of configurations to run before kill switching')
parser.add_argument('--fairness_metric', type=str, default='statistical_parity_difference',
                    help='Fairness Metric')
args = parser.parse_args()

SEED = 42
N_TRIALS = 200
if args.run_name == 'None':
    run_name = f"{args.dataset}_{args.privilege_mode}_{args.model}_{args.prepost}_{args.fairness_metric}"
else:
    run_name = args.run_name
run_path = f'results/{run_name}/{SEED}/'

# Define the SMAC Run with all the arguments
SMAC_RUN = ['python3', 'train.py',
    '--dataset', args.dataset,
    '--privilege_mode', args.privilege_mode,
    '--train_bs', '64',
    '--test_bs', '64',
    '--model', args.model,
    '--prepost', args.prepost,
    '--multi_objective',
    '--weighting', str(args.weighting),
    '--fairness_metric', args.fairness_metric,
    '--run_name', run_name,
    '--output_dir', 'results/',
    '--wall_time_limit', '1000000',
    '--n_trials', str(N_TRIALS),
    '--initial_n_configs', '10',
    '--min_budget', '2.5',
    '--max_budget', '10',
    '--eval_budget', '10',
    '--successive_halving_eta', '3',
    '--seed', str(SEED)
]

with open(f'logs/{run_name}.log', 'w') as f:
    process = subprocess.Popen(SMAC_RUN, stdout=f, stderr=f)
    pid = process.pid

print('SMAC Run: ', ' '.join(SMAC_RUN), 'started with PID:', pid)

NUM_RESTARTS = 0
while True:
    if os.path.exists(run_path):
        with open(os.path.join(run_path, 'runhistory.json'), 'r') as f:
            runhistory = json.load(f)
            print("CONFIGS FINISHED RUNNING: ", runhistory["stats"]["finished"])
            if runhistory["stats"]["finished"] >= (NUM_RESTARTS+1)*args.num_configs_switch:
                print('KILLING THE SMAC_RUN WITH PID:', pid)
                os.kill(pid, signal.SIGTERM)
                print('PROCESS KILLED')
                print('RESTARTING THE SMAC_RUN')
                with open(f'logs/{run_name}.log', 'a') as f:
                    process = subprocess.Popen(SMAC_RUN, stdout=f, stderr=f)
                    pid = process.pid
                print('SMAC Run: ', ' '.join(SMAC_RUN), 'continuing with PID:', pid)
                NUM_RESTARTS += 1
            if runhistory["stats"]["finished"] >= N_TRIALS:
                print('SMAC RUN FINISHED...')
                print('Exiting in 10 seconds...')
                time.sleep(10)
                break
    else:
        print(os.path.join(run_path, 'runhistory.json'), 'NOT FOUND...')
        print('WAITING FOR THE FILE TO BE CREATED')
    time.sleep(10)
