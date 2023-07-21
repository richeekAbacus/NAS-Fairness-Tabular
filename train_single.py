import os
import json
import torch
import argparse

from smac import Scenario
from smac.multi_objective.parego import ParEGO
from smac import MultiFidelityFacade as MFFacade
from smac.intensifier.hyperband import Hyperband

from ConfigSpace import Configuration

from utils import plot_pareto, get_fairness_obj, log_fairness_metrics
from dataloaders import get_adult_dataloaders, get_compas_dataloaders, get_acsincome_dataloaders
from models import (FTTransformerSearch, ResNetSearch, MLPSearch,
                    ResNetPreSearch, FTTransformerPreSearch,
                    ResNetPostSearch, FTTransformerPostSearch)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='adult')
parser.add_argument('--privilege_mode', type=str, default='sex')
parser.add_argument('--train_bs', type=int, default=64)
parser.add_argument('--test_bs', type=int, default=64)
parser.add_argument('--model', type=str, default='FTTransformer',
                    choices=['FTTransformer', 'ResNet', 'MLP',
                             'ResNetPre', 'FTTransformerPre',
                             'ResNetPost', 'FTTransformerPost'],
                    help="which model to use for NAS"),
parser.add_argument('--prepost', type=str, default="None",
                    help="Pre/Post-processing technique to use if such a `model` is chosen")
parser.add_argument('--multi_objective', action='store_true',
                    help="whether to use multi-objective optimization \
                        -> joint optimization of both accuracy and fairness")
parser.add_argument('--target', type=str, default='rev_acc',
                    choices=['rev_acc', 'fairness'],
                    help="which metric to use as the target for optimization")
parser.add_argument('--weighting', type=int, default=1,
                    help="amount of weighting to apply to fairness metric: 1 is no-weighting -> uses ParEgo")
parser.add_argument('--fairness_metric', type=str, default='statistical_parity_difference',
                    choices=['statistical_parity_difference',
                             'disparate_impact',
                             'average_odds_difference',
                             'average_abs_odds_difference',
                             'equal_opportunity_difference'],
                    help="which fairness metric to use for multi-objective optimization")
parser.add_argument('--use_advanced_num_embeddings', action='store_true')
parser.add_argument('--use_mlp', action='store_true')
parser.add_argument('--use_intersample', action='store_true')
parser.add_argument('--use_long_ffn', action='store_true')
parser.add_argument('--run_name', type=str, default='test')
parser.add_argument('--output_dir', type=str, default='results/')
parser.add_argument('--wall_time_limit', type=int, default=3600, help="in seconds")
parser.add_argument('--n_trials', type=int, default=15)
parser.add_argument('--initial_n_configs', type=int, default=5,
                    help="number of initial random configs to run")
parser.add_argument('--min_budget', type=float, default=2.5)
parser.add_argument('--max_budget', type=float, default=10)
parser.add_argument('--eval_budget', type=float, default=10)
parser.add_argument('--successive_halving_eta', type=int, default=3)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

#! TODO: Write a new dataloader for the adult dataset. This one drops a lot of columns!

DATA_FN_MAP = {
    'adult': get_adult_dataloaders,
    'compas': get_compas_dataloaders,
    'acs-income': get_acsincome_dataloaders
}

if __name__ == "__main__":
    print("LOG ARGS", "\n", args)

    if args.model == 'FTTransformer':
        model_search = FTTransformerSearch(args, DATA_FN_MAP[args.dataset],
                                           fairness_metric=args.fairness_metric if \
                                                           args.multi_objective else None)
    elif args.model == 'ResNet':
        model_search = ResNetSearch(args, DATA_FN_MAP[args.dataset],
                                    fairness_metric=args.fairness_metric if \
                                                    args.multi_objective else None)
    elif args.model == 'MLP':
        model_search = MLPSearch(args, DATA_FN_MAP[args.dataset],
                                 fairness_metric=args.fairness_metric if \
                                                 args.multi_objective else None)
    elif args.model == "ResNetPre":
        model_search = ResNetPreSearch(args, DATA_FN_MAP[args.dataset],
                                       fairness_metric=args.fairness_metric if \
                                                       args.multi_objective else None,
                                       preprocessing_method=args.prepost)
    elif args.model == "FTTransformerPre":
        model_search = FTTransformerPreSearch(args, DATA_FN_MAP[args.dataset],
                                              fairness_metric=args.fairness_metric if \
                                                              args.multi_objective else None,
                                              preprocessing_method=args.prepost)
    elif args.model == "ResNetPost":
        model_search = ResNetPostSearch(args, DATA_FN_MAP[args.dataset],
                                        fairness_metric=args.fairness_metric if \
                                                        args.multi_objective else None,
                                        postprocessing_method=args.prepost)
    elif args.model == "FTTransformerPost":
        model_search = FTTransformerPostSearch(args, DATA_FN_MAP[args.dataset],
                                               fairness_metric=args.fairness_metric if \
                                                               args.multi_objective else None,
                                               postprocessing_method=args.prepost)
    else:
        raise ValueError("Model not recognized!")

    print('Running NAS on %d GPUs'%torch.cuda.device_count())
    
    scenario = Scenario(
        model_search.configspace,
        name=args.run_name,
        output_directory=args.output_dir,
        # objectives=model_search.get_objectives,
        walltime_limit=args.wall_time_limit,  # After n seconds, we stop the hyperparameter optimization
        n_trials=args.n_trials,  # Evaluate max n different trials
        min_budget=args.min_budget,  # Train the model using a architecture configuration for at least n epochs
        max_budget=args.max_budget,  # Train the model using a architecture configuration for at most n epochs
        seed=args.seed,
        n_workers=torch.cuda.device_count(),
    )

    # We want to run n random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario, n_configs=args.initial_n_configs)
    intensifier = Hyperband(scenario, eta=args.successive_halving_eta,
                            incumbent_selection="highest_budget") # SuccessiveHalving can be used
    
    if args.weighting == 1:
        multi_objective_algorithm = ParEGO(scenario)
    else:
        assert args.weighting > 1
        print("Using weighted multi-objective optimization with weighting %d"%args.weighting)
        multi_objective_algorithm = MFFacade.get_multi_objective_algorithm(scenario, objective_weights=[1, args.weighting])

    optimvar = args.target if args.target == 'rev_acc' else args.target + '_obj' 

    smac = MFFacade(
        scenario,
        lambda config, seed, budget: model_search.train(config, seed, budget)[optimvar],
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=False,
    )

    # Init log files if they don't exist
    if not os.path.exists(os.path.join(args.output_dir, args.run_name, str(args.seed), 'stats.csv')):
        with open(os.path.join(args.output_dir, args.run_name, str(args.seed), 'stats.csv'), 'w') as f:
            f.write("Classification accuracy, Balanced classification accuracy, Mean Difference, Disparate impact, Equal opportunity difference, Average odds difference, Theil_index\n")
        with open(os.path.join(args.output_dir, args.run_name, str(args.seed), 'stats.json'), 'w') as f:
            json.dump({"data": []}, f)

    incumbents = smac.optimize()
    if isinstance(incumbents, Configuration):
        incumbents = [incumbents]
  
    print("\nBest found configurations: %s" % (incumbents))
    print("Found: ", len(incumbents), " configurations on the pareto front!")

    default_config = model_search.configspace.get_default_configuration()
    train_acc, val_acc, test_acc, val_class_metric, test_class_metric =\
        model_search.create_and_train_model_from_config(config=default_config, budget=args.eval_budget)
    print(f"\nDefault train score: {train_acc}")
    print(f"Default val score: {val_acc}")
    print(f"Default test score: {test_acc}")
    print(f"Default val fairness obj: {get_fairness_obj(val_class_metric, args.fairness_metric)}")
    print(f"Default test fairness obj: {get_fairness_obj(test_class_metric, args.fairness_metric)} \n\n")

    with open(os.path.join(args.output_dir, args.run_name, str(args.seed), 'incumbents.csv'), 'w') as f:
        f.write("Model, Classification accuracy, Balanced classification accuracy, Mean Difference, Disparate impact, Equal opportunity difference, Average odds difference, Theil_index\n")
        f.write('Default,' + ','.join(str(x) for x in log_fairness_metrics(test_class_metric)) + '\n')

    for idx, incumbent in enumerate(incumbents):
        print(f"\nIncumbent {idx}")
        train_acc, val_acc, test_acc, val_class_metric, test_class_metric =\
            model_search.create_and_train_model_from_config(config=incumbent, budget=args.eval_budget)
        print(f"\nIncumbent {idx}")
        print(f"\nIncumbent train score: {train_acc}")
        print(f"Incumbent val score: {val_acc}")
        print(f"Incumbent test score: {test_acc}")
        print(f"Incumbent val fairness obj: {get_fairness_obj(val_class_metric, args.fairness_metric)}")
        print(f"Incumbent test fairness obj: {get_fairness_obj(test_class_metric, args.fairness_metric)} \n\n")

        with open(os.path.join(args.output_dir, args.run_name, str(args.seed), 'incumbents.csv'), 'a') as f:
            f.write('Incumbent{},'.format(idx) +\
                    ','.join(str(x) for x in log_fairness_metrics(test_class_metric)) + '\n')

    if args.multi_objective:
        plot_pareto(smac, incumbents, args)
