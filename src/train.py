import os
import pickle5 as pickle
import sys

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from mlem.api import save

# load hyperparameters from an external file
params = yaml.safe_load(open("params.yaml"))["train"]

# validate that you have all the arguments needed for the experiment run
if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py features model\n")
    sys.exit(1)

# get the input location/directory for the data and the output location/directory for the model
input = sys.argv[1]
output = sys.argv[2]

# define the hyperparameter values that come from the external
seed = params["seed"]
n_est = params["n_est"]
min_split = params["min_split"]

# load the training data from the input location
with open(os.path.join(input, "test.pkl"), "rb") as fd:
    matrix = pickle.load(fd)

# separate the training data into labels and features
labels = matrix.iloc[:, 11].values
x = matrix.iloc[:,1:11].values

# validate the data you loaded and separated
sys.stderr.write("Input matrix size {}\n".format(matrix.shape))
sys.stderr.write("X matrix size {}\n".format(x.shape))
sys.stderr.write("Y matrix size {}\n".format(labels.shape))

# define the algorithm used for training
clf = RandomForestClassifier(
    n_estimators=n_est, min_samples_split=min_split, n_jobs=2, random_state=seed
)

# train the model
clf.fit(x, labels)

# output the model
with open(output, "wb") as fd:
    pickle.dump(clf, fd)
