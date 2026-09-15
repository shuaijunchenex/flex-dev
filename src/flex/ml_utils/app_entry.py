from __future__ import annotations
from abc import ABC
from typing import Any, Dict

from .console import console
from .filename_helper import FileNameHelper
from .string import String
from .config_loader import ConfigLoader
from .object_map import ObjectMap


class AppEntry(ABC, ObjectMap):
    """
    App entry class
    """
    
    # Static class variables
    __app_objects = ObjectMap()  # App objects map
    __app_config_define: dict = {}  # App config define dict
    _experiment_name: str = ""      # Auto-derived from config filename stem

    @staticmethod
    def set_app_object(key: Any, object: Any):
        AppEntry.__app_objects.set_object(key, object)

    @staticmethod
    def get_app_object(key: Any):
        return AppEntry.__app_objects.get_object(key)

    # ----------------------------------------------
    def __init__(self):
        ObjectMap.__init__(self)

    @property
    def app_objects(self):
        return AppEntry.__app_objects

    @property
    def app_config_define(self):
        return AppEntry.__app_config_define

    def run(self, training_rounds=50):
        """
        Run app
        """
        pass

    def load_app_config(self, app_config_define_file: str):
        """
        Load app config define file and parse it
        """
        if not FileNameHelper.exists(app_config_define_file):
            console.raise_exception(ValueError(f"App config define file '{app_config_define_file}' not exists."))

        AppEntry.__app_config_define = ConfigLoader.load(app_config_define_file)

        file_part = FileNameHelper.split(app_config_define_file)
        # Store experiment name so FedNodeVars.prepare_training_logger can auto-set prefix
        AppEntry._experiment_name = file_part.name
        self.__parse_app_config_define(file_part.folder, AppEntry.__app_config_define)
        return

    # def __parse_app_config_define(self, folder: str, config_define: dict) -> None:
    #     yaml_map = {}

    #     yaml_path = config_define.get("yaml_section_path", "")
        
    #     # Section: "yaml_section_files"
    #     for yaml_section_files in config_define["yaml_section_files"]:
    #         file_name = next(iter(yaml_section_files))
    #         name = yaml_section_files[file_name]

    #         # Check file exists
    #         fullname = FileNameHelper.combine(folder, yaml_path, file_name)
    #         if not FileNameHelper.exists(fullname):
    #             raise ValueError(f"yaml file not found '{fullname}'")

    #         # Empty name
    #         if String.is_none_or_empty(name):
    #             file_part = FileNameHelper.split(fullname)
    #             name = file_part.name

    #         config_dict = ConfigLoader.load(fullname)
    #         AppEntry.set_app_object(name, config_dict)
    #         yaml_map[name] = fullname

    #     # Section: "yaml_combination"
    #     yaml_combine = config_define["yaml_combination"]
    #     for cfg_name, combination in yaml_combine.items():
    #         if cfg_name == "yaml_section_path" or cfg_name == "yaml_section_files":
    #             continue

    #         # Existence check -- repeat not allowed
    #         if AppEntry.__app_objects.exists_object(cfg_name):
    #             raise ValueError(f"Combined yaml '{cfg_name}' already exists in app configs")

    #         combine_dict = {}
    #         if isinstance(combination, list):
    #             for name in combination:
    #                 config_dict = ConfigLoader.load(yaml_map[name])
    #                 combine_dict.update(config_dict)

    #         AppEntry.set_app_object(cfg_name, combine_dict)
    #     return

    def __parse_app_config_define(self, folder: str, config_define: dict) -> None:
        """
        Parse application configuration definition and register YAML sections and combinations.
        - Backward-compatible: supports both the "legacy" flat schema and the new "folder-per-section" schema.
        - Side effect: registers each loaded YAML dict into AppEntry under its alias name;
        then builds combined configs under names in `yaml_combination`.

        Expected schemas:

        Legacy schema (OLD):
        yaml_section_path: "./yamls/"
        yaml_section_files:
            - "section-general.yaml": general
            - "section-optimizer.yaml": optimizer
        yaml_combination:
            runner: [general, optimizer]

        New schema (NEW):
        yaml_folder_<section>_path: "./yamls/<section_dir>/"
        yaml_folder_<section>_files:
            - "<file_a>.yaml": <alias_a>
            - "<file_b>.yaml": <alias_b>
        yaml_combination:
            runner: [general_round30, training_logger, ...]

        Notes:
        - If alias is empty or None, we fallback to the filename stem.
        - We keep a map alias -> full path, so combinations can be (re)loaded from disk like before.
        - If a combination refers to an alias that wasn't registered, we raise an error.
        """
        yaml_map: Dict[str, str] = {}

        # -------------------------------
        # Helper functions
        # -------------------------------
        def _alias_or_stem(fullname: str, alias: str) -> str:
            """Return alias if provided; otherwise use filename stem."""
            if not String.is_none_or_empty(alias):
                return alias
            parts = FileNameHelper.split(fullname)  # expects .name without extension available
            return parts.name

        def _load_and_register(fullname: str, alias: str) -> None:
            """Load YAML at fullname, register into AppEntry with alias, and remember path in yaml_map."""
            if not FileNameHelper.exists(fullname):
                raise ValueError(f"yaml file not found '{fullname}'")
            config_dict = ConfigLoader.load(fullname)
            AppEntry.set_app_object(alias, config_dict)
            yaml_map[alias] = fullname

        # -------------------------------
        # Path + files loaders (OLD vs NEW)
        # -------------------------------
        if "yaml_section_files" in config_define:
            # ===== Legacy path =====
            yaml_path = config_define.get("yaml_section_path", "")

            for yaml_section_files in config_define["yaml_section_files"]:
                file_name = next(iter(yaml_section_files))
                alias = yaml_section_files[file_name]

                fullname = FileNameHelper.combine(folder, yaml_path, file_name)
                alias = _alias_or_stem(fullname, alias)
                _load_and_register(fullname, alias)

        else:
            # ===== New folder-per-section schema =====
            # Detect all keys like: yaml_folder_<section>_path  and yaml_folder_<section>_files
            # We build pairs on the fly to avoid hard-coding section names.
            # Example pair keys:
            #   yaml_folder_training_logger_path  <--->  yaml_folder_training_logger_files
            #   yaml_folder_optimizer_path        <--->  yaml_folder_optimizer_files
            new_keys = [k for k in config_define.keys() if k.startswith("yaml_folder_")]

            # Build map: section -> (path_key, files_key)
            section_pairs = {}
            for k in new_keys:
                if k.endswith("_path"):
                    section = k[len("yaml_folder_"):-len("_path")]
                    files_key = f"yaml_folder_{section}_files"
                    if files_key in config_define:
                        section_pairs[section] = (k, files_key)

            # Load each <section> block: iterate its files list (list of one-item dicts)
            for section, (path_key, files_key) in section_pairs.items():
                yaml_path = config_define.get(path_key, "")
                files_list = config_define.get(files_key, []) or []
                for item in files_list:
                    file_name = next(iter(item))
                    alias = item[file_name]

                    fullname = FileNameHelper.combine(folder, yaml_path, file_name)
                    alias = _alias_or_stem(fullname, alias)
                    # If alias already exists, keep parity with previous behavior: last write wins is OK,
                    # but we first check duplicate in combinations stage. Here we simply register.
                    _load_and_register(fullname, alias)

        # -------------------------------
        # Build combinations (same behavior as before)
        # -------------------------------
        yaml_combine = config_define.get("yaml_combination", {})
        for cfg_name, combination in yaml_combine.items():
            # Skip legacy meta keys if ever present in combinations (parity with old filter)
            if cfg_name in ("yaml_section_path", "yaml_section_files"):
                continue

            # Existence check -- repeat not allowed
            if AppEntry.__app_objects.exists_object(cfg_name):
                raise ValueError(f"Combined yaml '{cfg_name}' already exists in app configs")

            combine_dict: Dict[str, Any] = {}
            if isinstance(combination, list):
                for alias in combination:
                    # Ensure alias is known from the earlier registration phase
                    
                    if alias not in yaml_map:
                        # Keep explicit error to help diagnose mismatches (e.g., wrong alias in combination)
                        raise KeyError(
                            f"Combined yaml '{cfg_name}' references unknown alias '{alias}'. "
                            f"Known aliases: {sorted(yaml_map.keys())}"
                        )

                    if alias.startswith("general_round"):
                        # Extract training_rounds from alias like:
                        #   general_round10, general_round100, general_round100_cuda, ...
                        import re
                        m = re.match(r"general_round(\d+)", alias)
                        if m:
                            self.training_rounds = int(m.group(1))

                    config_dict = ConfigLoader.load(yaml_map[alias])
                    # Shallow update keeps last-in precedence (same as before)
                    combine_dict.update(config_dict)

            AppEntry.set_app_object(cfg_name, combine_dict)

        return

    
        # yaml_folder_aggreation_path: "./yamls/aggregation/"
        # yaml_folder_client_selection_path: "./yamls/client_selection/"
        # yaml_folder_dataset_loader_path: "./yamls/dataset_loader/"
        # yaml_folder_distribution_path: "./yamls/data_distribution/"
        # yaml_folder_loss_func_path: "./yamls/loss_func/"
        # yaml_folder_lora_path: "./yamls/lora/"
        # yaml_folder_nn_model_path: "./yamls/nn_model/"
        # yaml_folder_fl_nodes_path: "./yamls/fl_nodes/"
        # yaml_folder_optimizer_path: "./yamls/optimizer/"
        # yaml_folder_training_path: "./yamls/training/"
        # yaml_folder_client_strategy_path: "./yamls/client_strategy/"
        # yaml_folder_server_strategy_path: "./yamls/server_strategy/"
        # yaml_folder_runner_strategy_path: "./yamls/runner_strategy/"
        # yaml_folder_trainer_path: "./yamls/trainer/"
        # yaml_folder_training_logger_path: "./yamls/training_logger/"
        # yaml_folder_training_path: "./yamls/training/"
        # yaml_folder_general_path: "./yamls/general"
        # yaml_folder_rank_distribution_path: "./yamls/rank_distribution"

    # yaml_folder_aggregation_path = config_define.get("yaml_folder_aggregation_path", "")
    #     yaml_folder_client_selection_path = config_define.get("yaml_folder_client_selection_path", "")
    #     yaml_folder_dataset_loader_path = config_define.get("yaml_folder_dataset_loader_path", "")
    #     yaml_folder_distribution_path = config_define.get("yaml_folder_distribution_path", "")
    #     yaml_folder_loss_func_path = config_define.get("yaml_folder_loss_func_path", "")
    #     yaml_folder_lora_path = config_define.get("yaml_folder_lora_path", "")
    #     yaml_folder_nn_model_path = config_define.get("yaml_folder_nn_model_path", "")
    #     yaml_folder_fl_nodes_path = config_define.get("yaml_folder_fl_nodes_path", "")
    #     yaml_folder_optimizer_path = config_define.get("yaml_folder_optimizer_path", "")
    #     yaml_folder_training_path = config_define.get("yaml_folder_training_path", "")
    #     yaml_folder_client_strategy_path = config_define.get("yaml_folder_client_strategy_path", "")
    #     yaml_folder_server_strategy_path = config_define.get("yaml_folder_server_strategy_path", "")
    #     yaml_folder_runner_strategy_path = config_define.get("yaml_folder_runner_strategy_path", "")
    #     yaml_folder_trainer_path = config_define.get("yaml_folder_trainer_path", "")
    #     yaml_folder_training_logger_path = config_define.get("yaml_folder_training_logger_path", "")
    #     yaml_folder_general_path = config_define.get("yaml_folder_general_path", "")
    #     yaml_folder_rank_distribution_path = config_define.get("yaml_folder_rank_distribution_path", "")

