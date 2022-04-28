from pathlib import Path
from typing import Dict, List, Sequence, Union, NamedTuple
from pydantic import BaseModel
from strictyaml import YAML, load

# Project Directories
FILE_ROOT = Path(__file__).resolve().parent
ROOT = FILE_ROOT.parent
CONFIG_FILE_PATH = ROOT.parent / "config.yml"
DATASET_DIR = ROOT / "datasets"
TRAINED_MODEL_DIR = ROOT / "db"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str


class PreprocessingConfig(BaseModel):
    """
    All configuration relevant to model preprocessing.
    """

    user_freq: float
    validation_freq: float


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    data_dir_path: str
    data_version: str
    model_dir_path: str
    results_dir_path: str
    model_params: Dict[str, Dict[str, Union[int, List[int]]]]


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig
    preprocessing_config: PreprocessingConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        preprocessing_config=PreprocessingConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
