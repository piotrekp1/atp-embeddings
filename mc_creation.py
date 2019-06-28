import numpy as np
import pandas as pd

lines = open('data/dependencies_train').readlines()

data = []
for line in lines:
    conj, prems = line.split(':')
    prems = prems.rstrip().split()
    for prem in prems:
        data.append([conj, prem, len(prems)])
df = pd.DataFrame(data, columns=['conj', 'prem', 'num_prems'])

df['conj_prem_score'] = 1/df['num_prems']
df_scores = df.groupby(['conj', 'prem'], as_index=False)['conj_prem_score'].sum()

# we use softmax function for probabilities
df_scores['soft_score'] = np.exp(df_scores['conj_prem_score'])

prems_soft_scores_sums = df_scores.groupby('prem')['soft_score'].sum()
df_scores['prob'] = df_scores['soft_score'] / df_scores['prem'].map(prems_soft_scores_sums)

df_scores.to_csv('data/prems_mc.csv', index=False)