import json
import os
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Final, List, Optional, Union

import pandas as pd
import ConfigSpace as CS

import numpy as np
import pyarrow.parquet as pq  # type: ignore
import time

#DATA_DIR_NAME = os.path.join(os.environ["HOME"], "hpo_benchmarks")

DATA_DIR_NAME = os.path.join(os.getcwd(), "hpo_bench", "hpo_benchmarks")

#DATA_DIR_NAME = "hpo_benchmarks"
#DATA_DIR_NAME = os.path.join(os.getcwd(), "tpe_utils/hpo_benchmarks")
SEEDS: Final = [665, 1319, 7222, 7541, 8916]
VALUE_RANGES = json.load(open("hp_configurations/hpo_bench.json"))


class AbstractBench(metaclass=ABCMeta):
    _rng: np.random.RandomState
    _value_range: Dict[str, List[Union[int, float, str]]]
    dataset_name: str
    def __init__(self):
        self.all_results = []

    def reset_results(self):
        self.all_results = []

    def add_config(self, config):
        self.all_results += [config]

    def reseed(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)

    def _fetch_discrete_config_space(self) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters(
            [
                CS.UniformIntegerHyperparameter(name=name, lower=0, upper=len(choices) - 1)
                if not isinstance(choices[0], (str, bool))
                else CS.CategoricalHyperparameter(name=name, choices=[str(i) for i in range(len(choices))])
                for name, choices in self._value_range.items()
            ]
        )
        return config_space

    @property
    @abstractmethod
    def config_space(self) -> CS.ConfigurationSpace:
        raise NotImplementedError


class HPOBench(AbstractBench):
    def __init__(
        self,
        model_name: str,
        dataset_id: int
    ):
        super() .__init__()
        # https://ndownloader.figshare.com/files/30379005
        dataset_info = [
            ("credit_g", 31),
            ("vehicle", 53),
            ("kc1", 3917),
            ("phoneme", 9952),
            ("blood_transfusion", 10101),
            ("australian", 146818),
            ("car", 146821),
            ("segment", 146822),
        ]
        order_list_info = {
            "nn": ["alpha", "batch_size", "depth", "learning_rate_init", "width"],
            "rf": ["max_depth", "max_features", "min_samples_leaf", "min_samples_split"],
            "xgb": ["colsample_bytree", "eta", "max_depth", "reg_lambda"]
        }
        budget_tuple_info = {
            "nn": ("iter", 243),
            "rf": ("n_estimators", 512),
            "xgb": ("n_estimators", 2000)
        }

        # self.num_0 = 0
        dataset_name, dataset_id = dataset_info[dataset_id]
        self.dataset_name = '%s_%s' % (dataset_name, model_name)
        self.order_list  = order_list_info[model_name]
        budget_name, budget_value = budget_tuple_info[model_name]
        data_path         = os.path.join(DATA_DIR_NAME, "hpo-bench-%s" % model_name, str(dataset_id), f"{model_name}_{dataset_id}_data.parquet.gzip")
        db                = pd.read_parquet(data_path, filters=[(budget_name, "==", budget_value),('subsample', "==", 1.0)])
        self._db          = db.drop([budget_name, "subsample"], axis = 1)
        #self._rng         = np.random.RandomState(seed)
        self._value_range = VALUE_RANGES[f"hpo-bench-{model_name}"]

    def ordinal_to_real(self, config):
        return {key: self._value_range[key][config[key]] for key in config.keys()}

    def _search_dataframe(self, row_dict, df):
        # https://stackoverflow.com/a/46165056/8363967
        mask = np.array([True] * df.shape[0])

        for i, param in enumerate(df.drop(columns=["result"], axis=1).columns):
            mask *= df[param].values == row_dict[param]
        idx = np.where(mask)
        assert len(idx) == 1, 'The query has resulted into mulitple matches. This should not happen. ' \
                              f'The Query was {row_dict}'
        idx = idx[0][0]
        result = df.iloc[idx]["result"]
        return result
    
    def complete_call(self, config):
        #_config = config.copy()
        key_path = config.copy()
        loss      = []
        test_info = {}
        time_init = time.time()
        idx = 0
        for seed in SEEDS:
            key_path["seed"] = seed
            res = self._search_dataframe(key_path, self._db)
            # loss.append(1 - res["info"]['val_scores']['acc'])
            loss.append(res["info"]['val_scores']['acc'])
            for key in res["info"]['test_scores'].keys():
                if idx == 0:
                    test_info[key]  = res["info"]['test_scores'][key]* 1.0 / len(SEEDS)
                else:
                    test_info[key] += res["info"]['test_scores'][key] * 1.0 / len(SEEDS)
            key_path.pop("seed")
            idx += 1
        loss = np.mean(loss)
        time_final = time.time()
        test_info['generalization_score'] = test_info['acc']
        test_info['time_init']  = time_init
        test_info['time_final'] = time_final 
        test_info['score'] = loss
        # return loss, test_info
        return test_info

    def call_and_add_ordinal(self, config):
        loss, _ = self.complete_call(config)
        return loss

    def __call__(self, config):
        new_config = self.ordinal_to_real(config)
        loss, test_info = self.complete_call(new_config)
        test_info.update(new_config)
        self.add_config(test_info)
        return loss

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        return self._fetch_discrete_config_space()