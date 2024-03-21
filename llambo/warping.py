import pandas as pd
import numpy as np
from rich.logging import RichHandler
from rich.console import Console
import logging

logger = logging.getLogger(__name__)

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


def setup_logging(logger, log_name):
    # Create a console instance for rich log output. This could be omitted if you're okay with the default console.
    console = Console()

    # Log format for the file handler
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # File handler for writing logs to a file
    file_handler = logging.FileHandler(log_name, mode='w')
    file_handler.setFormatter(formatter)

    # Rich terminal handler for colorful and formatted terminal log display
    # Since we're using the rich default format, we don't set a formatter here
    stream_handler = RichHandler(console=console)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Set the log level
    logger.setLevel(logging.INFO)