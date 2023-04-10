import matplotlib.pyplot as plt
import pandas as pd
import classification_util as util
ConcatedCICI = pd.read_csv('/home/irteam/wendyunji-dcloud-dir/wendyunji/Dataset/MLAC-dataset/encodedwithdlabel_ConcatedCICI.csv')

del ConcatedCICI['Unnamed: 0']
del ConcatedCICI['attack_category']
del ConcatedCICI['label']

X_train, X_test, y_train, y_test = util.TrainTestSplit(ConcatedCICI)

feature_names = [f"feature {i}" for i in range(X_train.shape[1])]

from sklearn.ensemble import RandomForestClassifier

rforest = RandomForestClassifier()
rforest.fit(X_train, y_train)

import time
import numpy as np

start_time = time.time()
importances = rforest.feature_importances_
print(importances)
std = np.std([tree.feature_importances_ for tree in rforest.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

import pandas as pd

rforest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
rforest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
fig.savefig('230408_1.png')