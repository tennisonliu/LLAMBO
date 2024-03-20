import pandas as pd
import numpy as np

class NumericalTransformer:
    '''
    Perform warping/unwarping of search sapce for numerical hyperparameters.
    '''
    def __init__(self, hyperparameter_constraints: dict):
        self.hyperparameter_constraints = hyperparameter_constraints

    def warp(self, config: pd.DataFrame):
        config_ = config.copy()
        # iterate through columns of config
        assert len(config_.columns) == len(self.hyperparameter_constraints)
        for col in config_.columns:
            # if column is a hyperparameter
            if col in self.hyperparameter_constraints:
                constraint = self.hyperparameter_constraints[col]
                type, transform, _ = constraint
                if transform == 'log':
                    assert type in ['int', 'float']
                    config_[col] = np.log10(config_[col])
        return config_
    
    def unwarp(self, config: pd.DataFrame):
        config_ = config.copy()
        # iterate through columns of config
        assert len(config_.columns) == len(self.hyperparameter_constraints)
        for col in config_.columns:
            # if column is a hyperparameter
            if col in self.hyperparameter_constraints:
                constraint = self.hyperparameter_constraints[col]
                type, transform, _ = constraint
                if transform == 'log':
                    assert type in ['int', 'float']
                    config_[col] = 10 ** config_[col]
        return config_


class DictToObject:
    def __init__(self, dictionary):
        self.attrs = []
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # If the value is a dictionary, recursively convert it to an object
                value = DictToObject(value)
            setattr(self, key, value)
            self.attrs.append(key)