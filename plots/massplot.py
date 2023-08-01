import glob
import subprocess

for folder_path in glob.glob("../results/*"):
    foldername = folder_path.split("/")[-1]
    dataset, privilege_mode = foldername.split("_")[0:2]
    CMD = ["python3", "plot.py",
           "--dataset", dataset,
           "--privilege_mode", privilege_mode,
           "--rundir", folder_path + "/42/",]
    subprocess.Popen(CMD)
    print("GENERATED PLOST FOR: ", foldername)
