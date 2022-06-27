import pandas as pd
import numpy as np
import glob
import pickle
import os
import json

from sklearn.metrics import accuracy_score,f1_score
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default="test", help='Experiment name')

args = parser.parse_args()

folders = glob.glob('./{}_seed*'.format(args.exp))
results = []

for folder in folders:
  runs = glob.glob('{}/*'.format(folder))
  for run in runs:
    model_file = "{}/model.pkl".format(run)
    args_file = "{}/commandline_args.json".format(run)
    preds_file = "{}/prediction_TESTSET.csv".format(run)

    if os.path.exists(args_file):
      df = pd.read_csv(preds_file)
      acc = accuracy_score(df['truth'], df['prediction'])
      f1 = f1_score(df['truth'], df['prediction'], average="micro")
      
      with open(args_file, 'r') as f:
        configs = json.load(f)

      tmp = {}
      tmp['Model Name'] = run.replace(folder + '/','')
      tmp['Accuracy'] = acc
      tmp['F1'] = f1

      for k in configs:
        tmp["config_" + k] = configs[k]
        
      results.append(tmp)

df = pd.DataFrame.from_dict(results)

print(df)

df = df[['Model Name', 'Accuracy', 'F1', 'config_seed']]
df = df.groupby(['Model Name']).mean()[['Accuracy','F1']]
print(df.to_latex())

plt.scatter(df['Accuracy', 'F1'])
plt.show()