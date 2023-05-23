from __future__ import annotations

import torch
import argparse

from smac import Scenario
from smac.facade import AbstractFacade
from smac.multi_objective.parego import ParEGO
from smac import MultiFidelityFacade as MFFacade
from smac.intensifier.hyperband import Hyperband

from dataloaders import get_adult_dataloaders
from fttransformer_nas import FTTransformerSearch


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='adult')
parser.add_argument('--train_bs', type=int, default=64)
parser.add_argument('--test_bs', type=int, default=64)
parser.add_argument('--model', type=str, default='FTTransformer')
parser.add_argument('--multi_objective', action='store_true',
                    help="whether to use multi-objective optimization \
                        -> joint optimization of both accuracy and fairness")
parser.add_argument('--use_advanced_num_embeddings', action='store_true')
parser.add_argument('--use_mlp', action='store_true')
parser.add_argument('--use_intersample', action='store_true')
parser.add_argument('--use_long_ffn', action='store_true')
parser.add_argument('--run_name', type=str, default='test')
parser.add_argument('--output_dir', type=str, default='results/')
parser.add_argument('--wall_time_limit', type=int, default=3600, help="in seconds")
parser.add_argument('--n_trials', type=int, default=200)
parser.add_argument('--initial_n_configs', type=int, default=10,
                    help="number of initial random configs to run")
parser.add_argument('--min_budget', type=float, default=2.5)
parser.add_argument('--max_budget', type=float, default=10)
parser.add_argument('--eval_budget', type=float, default=10)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

#! TODO: Write a new dataloader for the adult dataset. This one drops a lot of columns!

DATA_FN_MAP = {
    'adult': get_adult_dataloaders
}

if __name__ == "__main__":
    if args.model == 'FTTransformer':
        model_search = FTTransformerSearch(args, DATA_FN_MAP[args.dataset],
                                           fairness_search=args.multi_objective)

    facades: list[AbstractFacade] = []
    
    print('Running NAS on %d GPUs'%torch.cuda.device_count())
    
    scenario = Scenario(
        model_search.configspace,
        name=args.run_name,
        output_directory=args.output_dir,
        objectives=model_search.get_objectives,
        walltime_limit=args.wall_time_limit,  # After n seconds, we stop the hyperparameter optimization
        n_trials=args.n_trials,  # Evaluate max n different trials
        min_budget=args.min_budget,  # Train the model using a architecture configuration for at least n epochs
        max_budget=args.max_budget,  # Train the model using a architecture configuration for at most n epochs
        seed=args.seed,
        n_workers=torch.cuda.device_count(),
    )    

    # We want to run n random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario, n_configs=args.initial_n_configs)
    # intensifier = Hyperband(scenario, incumbent_selection="highest_budget") # SuccessiveHalving can be used
    intensifier = MFFacade.get_intensifier(scenario, eta=2)
    multi_objective_algorithm = ParEGO(scenario)

    smac = MFFacade(
        scenario,
        model_search.train,
        initial_design=initial_design,
        multi_objective_algorithm=multi_objective_algorithm,
        intensifier=intensifier,
        overwrite=True,
    )
    incumbent = smac.optimize()

    print("\nBest found configuration: %s" % (incumbent))

    default_config = model_search.configspace.get_default_configuration()
    train_acc, val_acc, test_acc, val_spd, test_spd = model_search.create_and_train_model_from_config(config=default_config, budget=args.eval_budget)
    print(f"\nDefault train score: {train_acc}")
    print(f"Default val score: {val_acc}")
    print(f"Default test score: {test_acc}")
    print(f"Default val statistical parity difference: {val_spd}")
    print(f"Default test statistical parity difference: {test_spd}")

    train_acc, val_acc, test_acc, val_spd, test_spd = model_search.create_and_train_model_from_config(config=incumbent, budget=args.eval_budget)
    print(f"\nIncumbent train score: {train_acc}")
    print(f"Incumbent val score: {val_acc}")
    print(f"Incumbent test score: {test_acc}")
    print(f"Incumbent val statistical parity difference: {val_spd}")
    print(f"Incumbent test statistical parity difference: {test_spd}")
    
    print("\nSelected Model")
    print(incumbent)

    facades.append(smac)

    # plot_trajectory(facades, args.output_directory, args.seed, args.name)