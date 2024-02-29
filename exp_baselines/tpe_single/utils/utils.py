import json
import os
from argparse import ArgumentParser, Namespace
from enum import Enum
from logging import DEBUG, FileHandler, Formatter, Logger, basicConfig, getLogger
from typing import Any, Dict, List, Literal, Optional, Type, TypedDict, Union

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

from tpe.utils.constants import (
    HPType,
    NumericType,
    config2type,
    type2config,
)


bool_json = Literal["True", "False"]


class MetaInformation(TypedDict):
    lower: Optional[NumericType]  # The lower bound of parameters
    upper: Optional[NumericType]  # The upper bound of parameters
    log: Optional[Union[bool_json, bool]]  # scale: If True, log, otherwise uniform
    q: Optional[NumericType]  # The quantization parameter"


class ParameterSettings(TypedDict):
    param_type: Union[Type[int], Type[float], Type[str], Type[bool]]  # parameter type
    ignore: bool_json  # Whether we ignore this parameter or not
    lower: Optional[NumericType]  # The lower bound of parameters
    upper: Optional[NumericType]  # The upper bound of parameters
    log: Optional[Union[bool_json, bool]]  # scale: If True, log, otherwise uniform
    q: Optional[NumericType]  # The quantization parameter"
    sequence: Optional[List[NumericType]]  # The choices for numerical parameters.
    choices: Optional[List[str]]  # The choices for categorical parameters. Must be str for config space
    dataclass: Optional[str]  # The choices for categorical parameters in dataclass
    default_value: Optional[Union[NumericType, str]]  # The default value for this parameter
    meta: Optional[MetaInformation]  # Any information that is useful


def get_args_from_parser(Choices: Enum, opts: Dict) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--opt_name", choices=list(opts.keys()), default="tpe")
    parser.add_argument("--exp_id", type=int, default=0)
    parser.add_argument("--max_evals", type=int, default=100)

    default_choice = getattr(Choices, Choices._member_names_[0]).name
    parser.add_argument("--dataset", choices=list([c.name for c in Choices]), default=default_choice)  # type: ignore

    args = parser.parse_args()
    return args


def get_filename_from_args(bench_name: str, constraints: List[Enum], args: Namespace) -> str:
    name = os.path.join(bench_name, args.dataset, args.opt_name, f"{args.exp_id:0>3}")
    return name


def get_hyperparameter_module(hp_module_path: str) -> Any:
    depth = len(hp_module_path.split("."))
    return getattr(__import__(hp_module_path, fromlist=[""] * depth), "hyperparameters")


def get_hyperparameter(
    param_name: str,
    config_type: HPType,
    settings: ParameterSettings,
    hp_module_path: str,
) -> HPType:
    """
    Create a hyperparameter class by CSH.<config_type>

    Args:
        param_name (str): The name of hyperparameter
        config_type (HPType):
            The class of hyperparameter in the CSH package
        settings (ParameterSettings):
            Parameter settings from a json file

    Returns:
        hp (HPType):
            The hyperparameter for ConfigurationSpace
    """
    config_dict = {key: val for key, val in settings.items() if key not in ["ignore", "param_type", "dataclass"]}
    hp_module = get_hyperparameter_module(hp_module_path)

    if "dataclass" in settings.keys():
        assert isinstance(settings["dataclass"], str)
        choices = getattr(hp_module, settings["dataclass"])
        config_dict["choices"] = [c.name for c in choices]

    if "meta" in settings.keys():
        assert isinstance(config_dict["meta"], dict)  # <== MetaInformation
        config_dict["meta"]["log"] = eval(config_dict["meta"].get("log", False))

    hyperparameter = getattr(CSH, config_type)(name=param_name, **config_dict)

    return hyperparameter


def get_config_space(searching_space: Dict[str, ParameterSettings], hp_module_path: str) -> CS.ConfigurationSpace:
    """
    Create the config space by CS.ConfigurationSpace based on searching space dict

    Args:
        searching_space (Dict[str, ParameterSettings]):
            The dict of the pairs of a parameter name and the details
        hp_module_path (str): The path of the module describing the searching space

    Returns:
        cs (CS.ConfigurationSpace):
            The configuration space that includes the parameters specified in searching_space
    """

    settings_keys = set(ParameterSettings.__annotations__.keys())
    cs = CS.ConfigurationSpace()
    type_choices = ["int", "float", "bool", "str"]
    for param_name, settings in searching_space.items():
        if param_name.startswith("_") or settings["ignore"] == "True":
            continue

        if settings["param_type"] not in type_choices:
            raise ValueError(
                f"`param_type` in params.json must have one of {type_choices}, " f"but got {settings['param_type']}."
            )

        if len(set(settings.keys()) - settings_keys) >= 1:
            raise ValueError(f"params.json must not include the keys not specified in {settings_keys}.")

        if "log" in settings.keys():
            if settings["log"] not in bool_json.__args__:  # type: ignore
                raise ValueError("`log` in params.json must have either True or False, " f"but got {settings['log']}.")
            settings["log"] = eval(settings["log"])  # type: ignore

        if "sequence" in settings.keys():
            config_type = "OrdinalHyperparameter"
        else:
            config_type = type2config[eval(settings["param_type"])]  # type: ignore

        cs.add_hyperparameter(
            get_hyperparameter(
                param_name=param_name,
                config_type=config_type,
                settings=settings,
                hp_module_path=hp_module_path,
            )
        )

    return cs


def get_logger(file_name: str, logger_name: str, disable: bool = False) -> Logger:
    subdirs = "/".join(file_name.split("/")[:-1])
    os.makedirs(f"log/{subdirs}", exist_ok=True)

    file_path = f"log/{file_name}.log"
    fmt = "%(asctime)s [%(levelname)s/%(filename)s:%(lineno)d] %(message)s"

    if not disable:
        basicConfig(filename=file_path, format=fmt, level=DEBUG)

    logger = getLogger(logger_name)
    file_handler = FileHandler(file_path, mode="a")
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(Formatter(fmt))
    logger.propagate = False
    logger.addHandler(file_handler)

    return logger


def extract_hyperparameter(
    eval_config: Dict[str, Any],
    config_space: CS.ConfigurationSpace,
    searching_space: Dict[str, ParameterSettings],
    hp_module_path: str,
) -> Dict[str, Any]:
    """
    Extract categorical values from another module

    Args:
        eval_config (Dict[str, Any]):
            The configuration to evaluate.
            The categorical is provided by one of the key strings.
        config_space (CS.ConfigurationSpace):
            The configuration space
        searching_space (Dict[str, ParameterSettings]):
            Dict information taken from a prepared json file.
        hp_module_path (str):
            The path where the `hyperparameters.py` for the target
            objective exists.

    Returns:
        return_config (Dict[str, Any]):
            The configuration that is created by replacing
            all the categorical keys to the object of interests
            such as function
    """
    return_config = {}
    hp_module = get_hyperparameter_module(hp_module_path)
    for key, val in eval_config.items():
        hp_info = searching_space[key]
        if not isinstance(val, str):
            return_config[key] = val
            continue

        if "dataclass" in hp_info.keys():
            choices_class_name = hp_info["dataclass"]
            assert isinstance(choices_class_name, str)
            choices = getattr(hp_module, choices_class_name)
            return_config[key] = getattr(choices, val)
        elif hasattr(config_space.get_hyperparameter(key), "choices"):
            return_config[key] = eval(val)

    return return_config


def get_random_sample(
    hp_name: str,
    is_categorical: bool,
    is_ordinal: bool,
    rng: np.random.RandomState,
    config_space: CS.ConfigurationSpace,
) -> NumericType:
    """
    Random sample of a provided hyperparameter

    Args:
        config_space (CS.ConfigurationSpace): The searching space information
        is_categorical (bool): Whether this hyperparameter is categorical
        is_ordinal (bool): Whether this hyperparameter is ordinal
        hp_name (str): The names of a hyperparameter
        rng (np.random.RandomState): The random state for numpy

    Returns:
        value (NumericType): A sampled value before conversion
            * categorical -> index of a symbol
            * log -> a value in a log-scale
            * q -> no quantization here
    """
    config = config_space.get_hyperparameter(hp_name)

    if is_categorical:
        choices = config.choices
        sample = rng.randint(len(choices))
    elif is_ordinal:
        if config.meta is None:
            raise ValueError("The meta information of the ordinal hyperparameter must be provided")

        log = config.meta.get("log", False)
        seq = config.sequence
        sample = seq[rng.randint(len(seq))]
        sample = np.log(sample) if log else sample
    else:
        lb = np.log(config.lower) if config.log else config.lower
        ub = np.log(config.upper) if config.log else config.upper
        sample = rng.uniform() * (ub - lb) + lb

    return sample


def check_value_range(hp_name: str, config: CSH.Hyperparameter, val: NumericType) -> None:
    if val < config.lower or val > config.upper:
        raise ValueError(f"The sampled value for {hp_name} must be [{config.lower},{config.upper}], but got {val}.")


def revert_eval_config(
    eval_config: Dict[str, NumericType],
    config_space: CS.ConfigurationSpace,
    is_categoricals: Dict[str, bool],
    is_ordinals: Dict[str, bool],
    hp_names: List[str],
) -> Dict[str, Any]:
    """
    Revert the eval_config into the original value range.
    For example,
        * categorical: index of a symbol -> the symbol
        * log float: log(value) -> exp(log(value)) = value
        * quantized value: value -> steped by q, i.e. q, 2q, 3q, ...

    Args:
        eval_config (Dict[str, NumericType]): The configuration to evaluate and revert
        config_space (CS.ConfigurationSpace): The searching space information
        is_categoricals (Dict[str, bool]): Whether each hyperparameter is categorical
        is_ordinals (Dict[str, bool]): Whether each hyperparameter is ordinal
        hp_names (List[str]): The list of the names of hyperparameters

    Returns:
        reverted_eval_config (Dict[str, Any])
    """
    converted_eval_config: Dict[str, Any] = {}
    for hp_name in hp_names:
        is_categorical, is_ordinal = is_categoricals[hp_name], is_ordinals[hp_name]
        config = config_space.get_hyperparameter(hp_name)
        val = eval_config[hp_name]

        if is_categorical:
            converted_eval_config[hp_name] = config.choices[val]
        elif is_ordinal:
            if config.meta is None:
                raise ValueError("The meta information of the ordinal hyperparameter must be provided")

            log = config.meta.get("log", False)
            vals = np.log(config.sequence) if log else np.array(config.sequence)
            diff = np.abs(vals - val)
            converted_eval_config[hp_name] = config.sequence[diff.argmin()]
        else:
            dtype = config2type[config.__class__.__name__]
            q = config.q
            if config.log:
                val = np.exp(val)
            if q is not None or dtype is int:
                lb = config.lower
                q = 1 if q is None and dtype is int else q
                val = np.round((val - lb) / q) * q + lb

            check_value_range(hp_name=hp_name, config=config, val=val)

            converted_eval_config[hp_name] = dtype(val)

    return converted_eval_config


def store_results(
    best_config: Dict[str, np.ndarray],
    logger: Logger,
    observations: Dict[str, np.ndarray],
    file_name: str,
    requirements: Optional[List[str]] = None,
) -> None:

    logger.info(f"\nThe observations: {observations}")
    if requirements is None:
        save_observations(filename=file_name, observations=observations)

        subdirs = "/".join(file_name.split("/")[:-1])
        os.makedirs(f"incumbents/{subdirs}", exist_ok=True)
        with open(f"incumbents/{file_name}.json", mode="w") as f:
            json.dump(best_config, f, indent=4)
    else:
        required_observations = {k: v for k, v in observations.items() if k in requirements}
        save_observations(filename=file_name, observations=required_observations)


def save_observations(filename: str, observations: Dict[str, np.ndarray]) -> None:
    """
    Save the observations during the optimization procedure

    Args:
        filename (str): The name of the json file
        observations (Dict[str, np.ndarray]): The observations to save
    """
    subdirs = "/".join(filename.split("/")[:-1])
    os.makedirs(f"results/{subdirs}", exist_ok=True)

    with open(f"results/{filename}.json", mode="w") as f:
        json.dump({k: v.tolist() for k, v in observations.items()}, f, indent=4)
