# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module to deal with all matters relating to loading example data sets, which we tune ML models to.
"""
from enum import IntEnum, auto

import numpy as np
import pandas as pd  # only needed for csv reader, maybe try something else
from sklearn import datasets

DATA_LOADER_NAMES = ("breast", "digits", "iris", "wine", "diabetes")
SCORERS_CLF = ("nll", "acc")
SCORERS_REG = ("mae", "mse")

class ProblemType(IntEnum):
    """The different problem types we consider. Currently, just regression (`reg`) and classification (`clf`).
    """

    clf = auto()
    reg = auto()


DATA_LOADERS = {
    "digits": (datasets.load_digits, ProblemType.clf),
    "iris": (datasets.load_iris, ProblemType.clf),
    "wine": (datasets.load_wine, ProblemType.clf),
    "breast": (datasets.load_breast_cancer, ProblemType.clf),
    #"boston": (datasets.load_boston, ProblemType.reg),
    "diabetes": (datasets.load_diabetes, ProblemType.reg),
}

assert sorted(DATA_LOADERS.keys()) == sorted(DATA_LOADER_NAMES)

# Arguably, this could go in constants, but doesn't cause extra imports being here.
METRICS_LOOKUP = {ProblemType.clf: SCORERS_CLF, ProblemType.reg: SCORERS_REG}


def get_problem_type(dataset_name):
    """Determine if this dataset is a regression of classification problem.

    Parameters
    ----------
    dataset : str
        Which data set to use, must be key in `DATA_LOADERS` dict, or name of custom csv file.

    Returns
    -------
    problem_type : ProblemType
        `Enum` to indicate if regression of classification data set.
    """
    if dataset_name in DATA_LOADERS:
        _, problem_type = DATA_LOADERS[dataset_name]
        return problem_type

    # Maybe we can come up with a better system, but for now let's use a convention based on the naming of the csv file.
    if dataset_name.startswith("reg-"):
        return ProblemType.reg
    if dataset_name.startswith("clf-"):
        return ProblemType.clf
    assert False, "Can't determine problem type from dataset name."

def load_data(dataset_name, data_root=None):  # pragma: io
    """Load a data set and return it in, pre-processed into numpy arrays.

    Parameters
    ----------
    dataset : str
        Which data set to use, must be key in `DATA_LOADERS` dict, or name of custom csv file.
    data_root : str
        Root directory to look for all custom csv files. May be ``None`` for sklearn data sets.

    Returns
    -------
    data : :class:`numpy:numpy.ndarray` of shape (n, d)
        The feature matrix of the data set. It will be `float` array.
    target : :class:`numpy:numpy.ndarray` of shape (n,)
        The target vector for the problem, which is `int` for classification and `float` for regression.
    problem_type : :class:`bayesmark.data.ProblemType`
        `Enum` to indicate if regression of classification data set.
    """
    if dataset_name in DATA_LOADERS:
        loader_f, problem_type = DATA_LOADERS[dataset_name]
        data, target = loader_f(return_X_y=True)
    return data, target, problem_type
