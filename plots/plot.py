import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot Pareto front for NAS-Fairness-Tabular along with standard AIF360 baselines')
parser.add_argument('--rundir', type=str, default='test',
                    help='Directory of the run to plot')
parser.add_argument('--dataset', type=str, default='adult',
                    help='Dataset to load aif360 results from')
parser.add_argument('--privilege_mode', type=str, default='sex',
                    help='Privilege mode to load aif360 results from')
args = parser.parse_args()

MAP_NAMES = {
    'statistical_parity_difference': 'Mean Difference',
    'disparate_impact': 'Disparate Impact',
    'equal_opportunity_difference': 'Equal Opportunity Difference',
    'average_odds_difference': 'Average Odds Difference',
    'theil_index': 'Theil Index'
}

def get_pareto_set(data_points):
    # very slow implementation of pareto set
    pareto_set = set()
    for dp in data_points:
        dominated = False
        for i, ps in enumerate(data_points):
            if (dp[0] >= ps[0] and dp[1] > ps[1]) or (dp[0] > ps[0] and dp[1] >= ps[1]):
                dominated = True
                break
        if not dominated:
            pareto_set.add(dp)
    return pareto_set


# def plot_SMAC_pareto(source='runhistory.json'):
#     with open(os.path.join(args.rundir, source), 'r') as f:
#         runhistory = json.load(f)

#     data_points = []
#     for dp in runhistory['data']:
#         data_points.append((dp[4][0], dp[4][1]))

#     pareto_set = np.array([np.array(x) for x in get_pareto_set(data_points)])
#     pareto_set = pareto_set[np.argsort(pareto_set[:, 0])]
#     data_points = np.array(data_points)


#     costs_x, costs_y = data_points[:, 0], data_points[:, 1]
#     pareto_costs_x, pareto_costs_y = pareto_set[:, 0], pareto_set[:, 1]

#     plt.scatter(costs_x, costs_y, marker="x", label="Search space")
#     plt.scatter(pareto_costs_x, pareto_costs_y, marker="x", c='r', label="NAS Pareto front")
#     plt.step(
#         [pareto_costs_x[0]] + pareto_costs_x.tolist() + [np.max(costs_x)],  # We add bounds
#         [np.max(costs_y)] + pareto_costs_y.tolist() + [np.min(pareto_costs_y)],  # We add bounds
#         where="post",
#         linestyle=":",
#     )

def process_metrics(value, metric='statistical_parity_difference'):
    if metric == 'disparate_impact':
        return abs(1 - value)
    elif metric == 'statistical_parity_difference':
        return abs(value)
    elif metric == 'average_odds_difference':
        return abs(value)
    elif metric == 'average_abs_odds_difference':
        return abs(value)
    elif metric == 'equal_opportunity_difference':
        return abs(value)
    elif metric == 'theil_index':
        return abs(value)
    else:
        raise NotImplementedError

def plot_SMAC_pareto(metric='statistical_parity_difference'):
    with open(os.path.join(args.rundir, "stats.json"), 'r') as f:
        runhistory = json.load(f)

    data_points = []
    for dp in runhistory['data']:
        data_points.append((1-dp['accuracy'], process_metrics(dp[metric], metric)))
    
    pareto_set = np.array([np.array(x) for x in get_pareto_set(data_points)])
    pareto_set = pareto_set[np.argsort(pareto_set[:, 0])]
    data_points = np.array(data_points)


    costs_x, costs_y = data_points[:, 0], data_points[:, 1]
    pareto_costs_x, pareto_costs_y = pareto_set[:, 0], pareto_set[:, 1]

    plt.scatter(costs_x, costs_y, marker="x", label="Search space")
    plt.scatter(pareto_costs_x, pareto_costs_y, marker="x", c='r', label="NAS Pareto front")
    plt.step(
        [pareto_costs_x[0]] + pareto_costs_x.tolist() + [np.max(costs_x)],  # We add bounds
        [np.max(costs_y)] + pareto_costs_y.tolist() + [np.min(pareto_costs_y)],  # We add bounds
        where="post",
        linestyle=":",
    )


def plot_AIF360(metric='statistical_parity_difference'):
    csv = pd.read_csv(os.path.join("baselines", f"{args.dataset}_{args.privilege_mode}.csv"))
    csv['Name'] = csv['Method'] + "_" + csv['Model'].astype(str)

    names = csv['Name'].tolist()
    methods = csv['Method'].tolist()
    acc = csv['Classification Accuracy'].tolist()
    spd = [process_metrics(x, metric) for x in csv[MAP_NAMES[metric]].tolist()]

    colors = ['orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
              'yellow', 'magenta', 'black', 'darkgreen', 'darkblue']

    cc = 0
    mylabel = methods[0]
    for i in range(len(names)):
        if i>0 and methods[i] != methods[i-1]:
            cc += 1
            mylabel = methods[i]
        plt.scatter(1-acc[i], abs(spd[i]), marker="x", c=colors[cc], label=mylabel)
        mylabel = "_nolegend_"
    

def main():
    METRIC_NAMES = ['statistical_parity_difference', 'disparate_impact',
                    'equal_opportunity_difference', 'average_odds_difference', 'theil_index']
    for metric in METRIC_NAMES:
        plt.figure()
        plot_SMAC_pareto(metric)
        plot_AIF360(metric)
        plt.title(f"Pareto-Front-{args.dataset}-{args.privilege_mode}")
        plt.xlabel("1 - Accuracy")
        plt.ylabel(f"Fairness (Objective {MAP_NAMES[metric]})")
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        plt.savefig(f"pareto_front_{metric}.png", dpi=600, bbox_inches='tight')


if __name__ == '__main__':
    main()