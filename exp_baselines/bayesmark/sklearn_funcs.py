import warnings
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from bayesmark.data import METRICS_LOOKUP, ProblemType,  load_data
import pickle

# Choices used for test problems, there is some redundant specification with sklearn funcs file here
MODEL_NAMES = ("DT", "MLP-adam", "MLP-sgd", "RF", "SVM", "ada", "kNN", "lasso", "linear")
#DATA_LOADER_NAMES = ("breast", "digits", "iris", "wine", "boston", "diabetes")
DATA_LOADER_NAMES = ("breast", "digits", "iris", "wine", "diabetes")
CLF_LOADER_NAMES  = ("breast", "digits", "iris", "wine")
REG_LOADER_NAMES  = ["diabetes"]


SCORERS_CLF = ("nll", "acc")
SCORERS_REG = ("mae", "mse")
METRICS = tuple(sorted(SCORERS_CLF + SCORERS_REG))



CV_SPLITS = 5

# We should add cat variables into some of these configurations but a lot of
# the wrappers for the BO methods really have trouble with cat types.

# kNN
knn_cfg = {
    "n_neighbors": {"type": "int", "space": "linear", "range": (1, 25)},
    "p": {"type": "int", "space": "linear", "range": (1, 4)},
}

# SVM
svm_cfg = {
    "C": {"type": "real", "space": "log", "range": (1.0, 1e3)},
    "gamma": {"type": "real", "space": "log", "range": (1e-4, 1e-3)},
    "tol": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
}

# DT
dt_cfg = {
    "max_depth": {"type": "int", "space": "linear", "range": (1, 15)},
    "min_samples_split": {"type": "real", "space": "logit", "range": (0.01, 0.99)},
    "min_samples_leaf": {"type": "real", "space": "logit", "range": (0.01, 0.49)},
    "min_weight_fraction_leaf": {"type": "real", "space": "logit", "range": (0.01, 0.49)},
    "max_features": {"type": "real", "space": "logit", "range": (0.01, 0.99)},
    "min_impurity_decrease": {"type": "real", "space": "linear", "range": (0.0, 0.5)},
}

# RF
rf_cfg = {
    "max_depth": {"type": "int", "space": "linear", "range": (1, 15)},
    "max_features": {"type": "real", "space": "logit", "range": (0.01, 0.99)},
    "min_samples_split": {"type": "real", "space": "logit", "range": (0.01, 0.99)},
    "min_samples_leaf": {"type": "real", "space": "logit", "range": (0.01, 0.49)},
    "min_weight_fraction_leaf": {"type": "real", "space": "logit", "range": (0.01, 0.49)},
    "min_impurity_decrease": {"type": "real", "space": "linear", "range": (0.0, 0.5)},
}

# MLP with ADAM
mlp_adam_cfg = {
    "hidden_layer_sizes": {"type": "int", "space": "linear", "range": (50, 200)},
    "alpha": {"type": "real", "space": "log", "range": (1e-5, 1e1)},
    "batch_size": {"type": "int", "space": "linear", "range": (10, 250)},
    "learning_rate_init": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "tol": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "validation_fraction": {"type": "real", "space": "logit", "range": (0.1, 0.9)},
    "beta_1": {"type": "real", "space": "logit", "range": (0.5, 0.99)},
    "beta_2": {"type": "real", "space": "logit", "range": (0.9, 1.0 - 1e-6)},
    "epsilon": {"type": "real", "space": "log", "range": (1e-9, 1e-6)},
}

# MLP with SGD
mlp_sgd_cfg = {
    "hidden_layer_sizes": {"type": "int", "space": "linear", "range": (50, 200)},
    "alpha": {"type": "real", "space": "log", "range": (1e-5, 1e1)},
    "batch_size": {"type": "int", "space": "linear", "range": (10, 250)},
    "learning_rate_init": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "power_t": {"type": "real", "space": "logit", "range": (0.1, 0.9)},
    #"tol": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "momentum": {"type": "real", "space": "logit", "range": (0.001, 0.999)},
    #"validation_fraction": {"type": "real", "space": "logit", "range": (0.1, 0.9)},
}

# AdaBoostClassifier
ada_cfg = {
    "n_estimators": {"type": "int", "space": "linear", "range": (10, 100)},
    "learning_rate": {"type": "real", "space": "log", "range": (1e-4, 1e1)},
}

# lasso
lasso_cfg = {
    "C": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
    "intercept_scaling": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
}

# linear
linear_cfg = {
    "C": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
    "intercept_scaling": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
}

MODELS_CLF = {
    "kNN": (KNeighborsClassifier, {}, knn_cfg),
    "SVM": (SVC, {"kernel": "rbf", "probability": True}, svm_cfg),
    "DT": (DecisionTreeClassifier, {"max_leaf_nodes": None}, dt_cfg),
    "RF": (RandomForestClassifier, {"n_estimators": 10, "max_leaf_nodes": None}, rf_cfg),
    "MLP-adam": (MLPClassifier, {"solver": "adam", "early_stopping": True}, mlp_adam_cfg),
    "MLP-sgd": (
        MLPClassifier,
        {"solver": "sgd", "early_stopping": True, "learning_rate": "invscaling", "nesterovs_momentum": True, 'max_iter': 40},
        mlp_sgd_cfg,
    ),
    "ada": (AdaBoostClassifier, {}, ada_cfg),
    "lasso": (
        LogisticRegression,
        {"penalty": "l1", "fit_intercept": True, "solver": "liblinear", "multi_class": "ovr"},
        lasso_cfg,
    ),
    "linear": (
        LogisticRegression,
        {"penalty": "l2", "fit_intercept": True, "solver": "liblinear", "multi_class": "ovr"},
        linear_cfg,
    ),
}

# For now, we will assume the default is to go thru all classifiers
assert sorted(MODELS_CLF.keys()) == sorted(MODEL_NAMES)

ada_cfg_reg = {
    "n_estimators": {"type": "int", "space": "linear", "range": (10, 100)},
    "learning_rate": {"type": "real", "space": "log", "range": (1e-4, 1e1)},
}

lasso_cfg_reg = {
    "alpha": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
    "fit_intercept": {"type": "bool"},
    #"normalize": {"type": "bool"},
    "max_iter": {"type": "int", "space": "log", "range": (10, 5000)},
    "tol": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "positive": {"type": "bool"},
}

linear_cfg_reg = {
    "alpha": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
    "fit_intercept": {"type": "bool"},
    #"normalize": {"type": "bool"},
    "max_iter": {"type": "int", "space": "log", "range": (10, 5000)},
    "tol": {"type": "real", "space": "log", "range": (1e-4, 1e-1)},
}

MODELS_REG = {
    "kNN": (KNeighborsRegressor, {}, knn_cfg),
    "SVM": (SVR, {"kernel": "rbf"}, svm_cfg),
    "DT":  (DecisionTreeRegressor, {"max_leaf_nodes": None}, dt_cfg),
    "RF":  (RandomForestRegressor, {"n_estimators": 10, "max_leaf_nodes": None}, rf_cfg),
    "MLP-adam": (MLPRegressor, {"solver": "adam", "early_stopping": True}, mlp_adam_cfg),
    "MLP-sgd": (
        MLPRegressor,  # regression crashes often with relu
        {
            "activation": "tanh",
            "solver": "sgd",
            "early_stopping": True,
            "learning_rate": "invscaling",
            "nesterovs_momentum": True,
            "max_iter": 40
        },
        mlp_sgd_cfg,
    ),
    "ada": (AdaBoostRegressor, {}, ada_cfg_reg),
    "lasso": (Lasso, {}, lasso_cfg_reg),
    "linear": (Ridge, {"solver": "auto"}, linear_cfg_reg),
}

def seed_everything():
    seed = 0
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# If both classifiers and regressors match MODEL_NAMES then the experiment
# launcher can simply go thru the cartesian product and do all combos.
assert sorted(MODELS_REG.keys()) == sorted(MODEL_NAMES)


class TestFunction(ABC):
    """Abstract base class for test functions in the benchmark. These do not need to be ML hyper-parameter tuning.
    """

    def __init__(self):
        """Setup general test function for benchmark. We assume the test function knows the meta-data about the search
        space, but is also stateless to fit modeling assumptions. To keep stateless, it does not do things like count
        the number of function evaluations.
        """
        # This will need to be set before using other routines
        self.api_config = None

    @abstractmethod
    def evaluate(self, params):
        """Abstract method to evaluate the function at a parameter setting.
        """

    def get_api_config(self):
        """Get the API config for this test problem.

        Returns
        -------
        api_config : dict(str, dict(str, object))
            The API config for the used model. See README for API description.
        """
        assert self.api_config is not None, "API config is not set."
        return self.api_config

class CustomForward:
    def __init__(self, func_to_evaluate):
        self.all_results          = []
        self.func_to_evaluate = func_to_evaluate

    def reset_results(self):
        self.all_results = []

    def add_config(self, config):
        self.all_results += [config]

    def __call__(self, config):
        results, add_metrics = self.func_to_evaluate(config)
        self.add_config(add_metrics)
        return results

class SklearnModelCustom(TestFunction):
    """Test class for sklearn classifier/regressor CV score objective functions.
    """

    # Map our short names for metrics to the full length sklearn name
    _METRIC_MAP = {
        "nll": "neg_log_loss",
        "acc": "accuracy",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
    }

    _METRIC_MAP_NAME = {
        "nll": "cross-entropy",
        "acc": "accuracy",
        "mae": "mean absolute error",
        "mse": "mean squared error",
    }

    _LIST_MODELS_REG = {
        "kNN": "sklearn.neighbors.KNeighborsRegressor",
        "SVM": "sklearn.svm.SVR",
        "DT": "sklearn.tree.DecisionTreeRegressor",
        "RF": "sklearn.ensemble.RandomForestRegressor",
        "MLP-adam": "sklearn.neural_network.MLPRegressor",
        "MLP-sgd": "sklearn.neural_network.MLPRegressor",
        "ada": "sklearn.ensemble.AdaBoostRegressor",
        "lasso": "sklearn.linear_model.Lasso",
        "linear": "sklearn.linear_model.Ridge"
    }

    _LIST_MODELS_CLF = {
        "kNN": "sklearn.neighbors.KNeighborsClassifier",
        "SVM": "sklearn.svm.SVC",
        "DT": "sklearn.tree.DecisionTreeClassifier",
        "RF": "sklearn.ensemble.RandomForestClassifier",
        "MLP-adam": "sklearn.neural_network.MLPClassifier",
        "MLP-sgd": "sklearn.neural_network.MLPClassifier",
        "ada": "sklearn.ensemble.AdaBoostClassifier",
        "lasso": "sklearn.linear_model.LogisticRegression",
        "linear": "sklearn.linear_model.LogisticRegression"
    }

    _LIST_MODELS_REG2 = {
        "kNN": "K Nearest Neighbor",
        "SVM": "SVM",
        "DT": "DecisionTree",
        "RF": "RandomForest",
        "MLP-adam": "MLP",
        "MLP-sgd": "MLP",
        "ada": "AdaBoost",
        "lasso": "LogisticRegression",
        "linear": "LogisticRegression"
    }

    _LIST_MODELS_CLF2 = {
        "kNN": "K Neighbors Neightboar",
        "SVM": "SVM",
        "DT": "DecisionTree",
        "RF": "RandomForest",
        "MLP-adam": "MLP",
        "MLP-sgd": "MLP",
        "ada": "AdaBoost",
        "lasso": "LogisticRegression",
        "linear": "LogisticRegression"
    }

    _LIST_CLF_MAP = [
        "neg_log_loss",
        "accuracy",
        "balanced_accuracy",
        "f1_micro",
        "roc_auc"
    ]

    _LIST_REG_MAP = [
       "neg_mean_absolute_error",
       "neg_mean_squared_error"
    ]

    _CUSTOM_DATASET = [
        "maggic", "seer", "cutract", 'Rosenbrock', 'Griewank', 'KTablet'
    ]


    _LOG_DATASET = [
        'Rosenbrock', 'Griewank', 'KTablet'
    ]

    def __init__(self, model, dataset, metric, shuffle_seed=0, data_root=None):
        """Build class that wraps sklearn classifier/regressor CV score for use as an objective function.

        Parameters
        ----------
        model : str
            Which classifier to use, must be key in `MODELS_CLF` or `MODELS_REG` dict depending on if dataset is
            classification or regression.
        dataset : str
            Which data set to use, must be key in `DATA_LOADERS` dict, or name of custom csv file.
        metric : str
            Which sklearn scoring metric to use, in `SCORERS_CLF` list or `SCORERS_REG` dict depending on if dataset is
            classification or regression.
        shuffle_seed : int
            Random seed to use when splitting the data into train and validation in the cross-validation splits. This
            is needed in order to keep the split constant across calls. Otherwise there would be extra noise in the
            objective function for varying splits.
        data_root : str
            Root directory to look for all custom csv files.
        """
        TestFunction.__init__(self)
        seed_everything()
        if dataset not in SklearnModelCustom._CUSTOM_DATASET:
            data, target, problem_type = load_data(dataset, data_root=data_root)
        else:
            file = open("custom_dataset/%s.pickle" % dataset,'rb')
            object_file = pickle.load(file)
            file.close()
            self.data_X  = object_file['train_x']
            self.data_Xt = object_file['test_x']
            self.data_y  = object_file['train_y']
            self.data_yt = object_file['test_y']
            data         = np.concatenate([self.data_X, self.data_Xt])
            target       = np.concatenate([self.data_y, self.data_yt])
            if dataset not in SklearnModelCustom._LOG_DATASET:
                problem_type = ProblemType.clf
            else:
                problem_type = ProblemType.reg

        assert problem_type in (ProblemType.clf, ProblemType.reg)
        self.is_classifier = problem_type == ProblemType.clf

        # Do some validation on loaded data
        assert isinstance(data, np.ndarray)
        assert isinstance(target, np.ndarray)
        assert data.ndim == 2 and target.ndim == 1
        assert data.shape[0] == target.shape[0]
        assert data.size > 0
        assert data.dtype == np.float_
        assert np.all(np.isfinite(data))  # also catch nan
        assert target.dtype == (np.int_ if self.is_classifier else np.float_)
        assert np.all(np.isfinite(target))  # also catch nan

        model_lookup = MODELS_CLF if self.is_classifier else MODELS_REG
        base_model, fixed_params, api_config = model_lookup[model]

        # New members for model
        self.base_model   = base_model
        self.fixed_params = fixed_params
        self.api_config   = api_config

        self.data   = data
        self.data   = np.nan_to_num(data, nan=0)
        self.target = target
        self.dataset_name = dataset

        if dataset not in SklearnModelCustom._CUSTOM_DATASET:
            # Always shuffle your data to be safe. Use fixed seed for reprod.
            self.data_X, self.data_Xt, self.data_y, self.data_yt = train_test_split(
                data, target, test_size=0.2, random_state = 0, shuffle = True
            )
        
        if not self.is_classifier:
            mean_train_y   = self.data_y.mean()
            std_train_y    = self.data_y.std()
            self.data_y    = (self.data_y  - mean_train_y) / std_train_y
            self.data_yt   = (self.data_yt - mean_train_y) / std_train_y

        assert metric in METRICS, "Unknown metric %s" % metric
        assert metric in METRICS_LOOKUP[problem_type], "Incompatible metric %s with problem type %s" % (
            metric,
            problem_type,
        )

        self.path_name    = 'metric_%s_data_%s_model_%s' % (metric, dataset, model)
        self.metric_name  = SklearnModelCustom._METRIC_MAP_NAME[metric]
        self.model_name   = SklearnModelCustom._LIST_MODELS_CLF[model] if self.is_classifier else SklearnModelCustom._LIST_MODELS_REG[model]
        self.model_name2  = SklearnModelCustom._LIST_MODELS_CLF2[model] if self.is_classifier else SklearnModelCustom._LIST_MODELS_REG2[model]

        self.scorer       = get_scorer(SklearnModelCustom._METRIC_MAP[metric])

        if dataset in SklearnModelCustom._LOG_DATASET:
            self.apply_log = True
        else:
            self.apply_log = False

    def obtain_evaluate(self, func_to_evaluate):
        return CustomForward(func_to_evaluate)

    def get_config_space(self):
        import ConfigSpace as CS
        config = self.api_config
        CONFIG_SPACE = CS.ConfigurationSpace()
        order_list   = []
        for key in config.keys():
            is_log = True if ('space' in config[key].keys() and config[key]['space'] in ['log', 'logit']) else False
            if config[key]['type'] == 'real':
                CONFIG_SPACE.add_hyperparameters([CS.UniformFloatHyperparameter(name=key,
                                                lower = config[key]['range'][0], upper = config[key]['range'][1], log = is_log)])
            elif config[key]['type'] == 'int':
                CONFIG_SPACE.add_hyperparameters([CS.UniformIntegerHyperparameter(name=key,
                                                lower = config[key]['range'][0], upper = config[key]['range'][1], log = is_log)])
            elif config[key]['type'] == 'bool':
                CONFIG_SPACE.add_hyperparameters([CS.CategoricalHyperparameter(key, [False, True])])
            order_list += [key]
        return CONFIG_SPACE, order_list

    def compute_clf_metrics(self, y_pred, y_true, y_pred_proba = None):
        from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
        additional_metrics = {}
        additional_metrics['f1_score']          = f1_score(y_true, y_pred, average = 'micro')
        additional_metrics['accuracy']          = accuracy_score(y_true, y_pred)
        additional_metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        # if y_pred_proba is not None:
        #     additional_metrics['cross_entropy'] = log_loss(y_true, y_pred_proba)
        #     if y_pred_proba.shape[1] == 2:
        #         additional_metrics['roc_auc']       = roc_auc_score(y_true, y_pred_proba[:, 1], average= 'micro', multi_class='ovr')
        #     else:
        #         additional_metrics['roc_auc']       = roc_auc_score(y_true, y_pred_proba, average= 'micro', multi_class='ovr')
        return additional_metrics

    def compute_reg_metrics(self, y_pred, y_true):
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        additional_metrics = {}
        additional_metrics['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)
        additional_metrics['mean_squared_error']  = mean_squared_error(y_true, y_pred)
        return additional_metrics

    def get_task_dict(self):
        def update_statistics(this_dict, X, Y, categorical_indicator):
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
            import pandas as pd
            import numpy as np

            label_encoder = LabelEncoder()
            # Fit and transform the target column
            X_new                 = pd.DataFrame(X)
            Y_new                 = pd.DataFrame(label_encoder.fit_transform(Y)).astype(float)
            numerical_indicator   = [not value for value in categorical_indicator]
            #X_new                 = X_new.reset_index()
            #X_new                 = X_new.reset_index().drop(X_new.columns[0], axis=1)

            categorical_features  = X_new.columns[categorical_indicator]
            numerical_features    = X_new.columns[numerical_indicator]

            X_encoded    = pd.get_dummies(X_new, columns = categorical_features)
            all_columns  = list(X_encoded.columns)
            Y_corr_abs   = X_encoded.corrwith(Y_new[0]).abs()
            X_corr_abs   = X_encoded.corr().abs()

            all_y_corr   = np.array(Y_corr_abs)
            num_y_corr   = len(all_y_corr)
            all_y_corr   = all_y_corr[~np.isnan(all_y_corr)]

            all_y_corr_pass = all_y_corr[all_y_corr > 0.5]
            num_y_corr_pass = len(all_y_corr_pass) 

            import numpy as np
            all_x_corr = []
            all_names  = []
            for this_column in all_columns:
                    all_rest_columns = all_columns[all_columns.index(this_column) + 1: ]
                    for this_row in all_rest_columns:
                            all_x_corr  += [X_corr_abs[this_column][this_row]]
                            all_names += [(this_column, this_row)]
            all_x_corr  = np.array(all_x_corr)
            all_names   = np.array(all_names) 

            num_x_corr      = len(all_x_corr)
            # Use boolean indexing to remove NaN values
            all_names       = all_names[~np.isnan(all_x_corr)]
            all_x_corr      = all_x_corr[ ~np.isnan(all_x_corr)]

            all_x_corr_pass = all_x_corr[all_x_corr > 0.5]
            num_x_corr_pass = len(all_x_corr_pass)

            skew     = list(X_new[numerical_features].skew())
            kurtosis = list(X_new[numerical_features].kurtosis())
            this_dict['skew']            = skew
            this_dict['kurtosis']        = kurtosis
            this_dict['num_y_corr']      = num_y_corr
            this_dict['num_x_corr']      = num_x_corr
            this_dict['num_y_corr_pass'] = num_y_corr_pass
            this_dict['num_x_corr_pass'] = num_x_corr_pass
            return this_dict
            
        def obtain_categorical_columns(this_df):
            categorical_features = []   
            threshold = 40  # You can adjust this threshold based on your dataset
            # Loop through each column in the dataset
            for column in this_df.columns:
                unique_values = this_df[column].nunique()
                # Check if the number of unique values is below the threshold
                if unique_values <= threshold:
                    #categorical_features.append(column)
                    categorical_features.append(True)
                else:
                    categorical_features.append(False)
            return categorical_features

        X, Y = self.data, self.target
        #task.target_name, dataset.name
        #categorical_indicator      = obtain_categorical_columns(pd.DataFrame(X))
        categorical_indicator      = [False] * X.shape[1]
        task_dict = {}
        task_dict['model']         = self.model_name
        task_dict['metric']        = self.metric_name
        task_dict['task']          = "classification"  if self.is_classifier else "regression"
        task_dict['num_samples']   = X.shape[0]
        task_dict['num_feat']      = X.shape[-1]
        task_dict['num_feat_cont'] = X.shape[-1] - np.sum(categorical_indicator)
        task_dict['num_feat_cat']  = np.sum(categorical_indicator)
        task_dict['num_by_class']  = list(pd.DataFrame(Y).value_counts())
        task_dict = update_statistics(task_dict, self.data_X, self.data_y, categorical_indicator)
        return task_dict

    def get_task_context(self):
        task_dict = self.get_task_dict()
        all_var = ['model', 'task', 'tot_feats', 'cat_feats', 'num_feats', 'n_classes', 'metric', 'num_samples']
        task_context = {}
        task_context['model']       = self.model_name2
        task_context['task']        = task_dict['task']
        task_context['tot_feats']   = task_dict['num_feat']
        task_context['cat_feats']   = task_dict['num_feat_cat']
        task_context['num_feats']   = task_dict['num_feat_cont']
        task_context['n_classes']   = len(task_dict['num_by_class'] )
        task_context['metric']      = 'accuracy' if task_dict['task'] == 'classification' else "neg_mean_squared_error"
        task_context['num_samples'] = self.data_X.shape[0]
        return task_context

    def obtain_pred(self, clf, X):
        try:
            y_pred_proba = clf.predict_proba(X)
            y_pred       = y_pred_proba.argmax(-1)
        except:
            y_pred_proba = None
            y_pred       = clf.predict(X)
        return y_pred, y_pred_proba

    def logit(self, p):
        """
        Transform a probability value 'p' to the log-odds scale.
        """
        return np.log(p / (1 - p + + 1e-8) + 1e-8) 

    def inverse_logit(self, log_odds):
        """
        Transform a value from the log-odds scale to the probability scale.
        """
        odds = np.exp(log_odds)
        return odds / (1 + odds)

    def evaluate(self, params):
        """Evaluate the sklearn CV objective at a particular parameter setting.

        Parameters
        ----------
        params : dict(str, object)
            The varying (non-fixed) parameter dict to the sklearn model.

        Returns
        -------
        cv_loss : float
            Average loss over CV splits for sklearn model when tested using the settings in params.
        """

        seed_everything()
        params = dict(params)  # copy to avoid modification of original
        params = {key: params[key] if params[key] != 'True' and params[key] != 'False' else params[key] == 'True'\
                                                                                            for key in params.keys()}
        params.update(self.fixed_params)  # add in fixed params
        if not self.model_name == "sklearn.svm.SVR":
            params.update({'random_state': 0})
        # now build the skl object
        
        clf = self.base_model(**params)

        assert np.all(np.isfinite(self.data_X)), "all features must be finite"
        assert np.all(np.isfinite(self.data_y)), "all targets must be finite"

        #print(params)
        # Do the x-val, ignore user warn since we expect BO to try weird stuff
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            S = cross_val_score(clf, self.data_X, self.data_y, scoring=self.scorer, cv=CV_SPLITS)
        # Take the mean score across all x-val splits
        cv_score = np.mean(S)


        # Now let's get the generalization error for same hypers
        clf = self.base_model(**params)
        clf.fit(self.data_X, self.data_y)

        # Predict on test
        if self.is_classifier:
            y_pred, y_pred_proba = self.obtain_pred(clf, self.data_Xt)
            additional_metrics   = self.compute_clf_metrics(y_pred, self.data_yt, y_pred_proba = y_pred_proba)
        else:
            #y_pred, y_pred_proba = self.obtain_pred(clf, self.data_Xt)
            y_pred               = clf.predict(self.data_Xt)
            additional_metrics   = self.compute_reg_metrics(y_pred, self.data_yt)

        generalization_score = self.scorer(clf, self.data_Xt, self.data_yt)
        #generalization_score = self.scorer._score_func(self.data_yt, y_pred)

        # get_scorer makes everything a score not a loss, so we need to negate to get the loss back
        cv_loss = -cv_score
        assert np.isfinite(cv_loss), "loss not even finite"
        generalization_loss = -generalization_score
        assert np.isfinite(generalization_loss), "loss not even finite"

        # Unbox to basic float to keep it simple
        cv_loss = cv_loss.item()
        assert isinstance(cv_loss, float)
        generalization_loss = generalization_loss.item()
        assert isinstance(generalization_loss, float)


        additional_metrics['generalization_score'] = generalization_score

        additional_metrics['obj_loss']             = cv_loss
        additional_metrics.update(params)

        return cv_loss, additional_metrics