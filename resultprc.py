# Process the results of NAS-Fairness-Tabular

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="results/test13/42/runhistory.json",\
                    help="path to the runhistory file")
parser.add_argument("--fairness_cutoff", type=float, default=0.1,\
                    help="fairness cutoff")
parser.add_argument("--accuracy_cutoff", type=float, default=0.1,\
                    help="accuracy cutoff")

args = parser.parse_args()

with open(args.path, "r") as f:
    data = json.load(f)

runs_selected = []
for run in data["data"]:
    acc, fairness_metric = run[4][0], run[4][1]
    # work under the assumption that fairness metrics -> smaller is better
    if 1 - acc > args.accuracy_cutoff and fairness_metric < args.fairness_cutoff:
        runs_selected.append(run[0])
        print("Run: {}, Accuracy: {}, Fairness: {}".format(run[0], acc, fairness_metric))
print("\n")

for run in runs_selected:
    print("Config: {}".format(run))
    print("Config: {}".format(data["configs"][str(run)]))
    print("\n")