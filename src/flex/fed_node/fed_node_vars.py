from __future__ import annotations
from typing import Any

import torch.nn as nn
import copy
from ..ml_data_loader.dataset_loader_util import DatasetLoaderUtil
from .fed_node_event_args import FedNodeEventArgs
from ..ml_utils import TrainingLogger, EventHandler, console, String, ObjectMap, KeyValueArgs
from ..ml_utils.model_utils import ModelUtils
from ..ml_models import NNModelFactory
from ..ml_data_loader import DatasetLoaderArgs, DatasetLoaderFactory, DatasetLoader
from ..ml_algorithms import LossFunctionBuilder, OptimizerBuilder, TokenizerBuilder
from ..fl_algorithms import FedClientSelectorFactory, FedClientSelector, FedClientSelectorArgs, FedAggregatorArgs, FedAggregatorFactory
from ..ml_data_process import DataDistribution
from ..fed_strategy.strategy_factory import StrategyFactory
from ..fl_algorithms import FedClientSelectorFactory
from ..model_trainer.model_evaluator import ModelEvaluator
from ..model_trainer.model_trainer_factory import ModelTrainerFactory
from ..model_trainer.model_trainer_args import ModelTrainerArgs

class FedNodeVars(ObjectMap, EventHandler, KeyValueArgs):
    """
    Fed node variables
    """

    # NOTICE: Static share model
    share_model: nn.Module|None = None
    share_vocab = None

    def __init__(self, config_dict: dict|None = None, is_clone_dict:bool = False):
        EventHandler.__init__(self)
        ObjectMap.__init__(self)
        KeyValueArgs.__init__(self, config_dict, is_clone_dict)

        # Computation device (default: cpu)
        self.device = "cpu"
        if config_dict is not None and "general" in config_dict:
            self.device = config_dict["general"].get("device", "cpu")

        # Variables owner node list. One for normal, more means var owned by more than one node
        self.__owner_nodes: list = []
        self.__init_vars()

        # Declare event
        self.declare_events("on_prepare_data_loader", "on_prepare_data_distribution", "on_prepare_data_handler", "on_prepare_model",
                            "on_prepare_optimizer", "on_prepare_loss_func", "on_prepare_client_selection", "on_prepare_trainer",
                            "on_prepare_aggregation", "on_prepare_strategy", "on_prepare_extractor", "on_prepare_training_logger", 
                            "on_prepare_lora_inference_model", "on_prepare_tokenizer", "on_prepare_vocab")
        return

    @property
    def config_dict(self) -> dict: return self.key_value_dict.dict

    @property
    def owner_nodes(self):
        return self.__owner_nodes

    @owner_nodes.setter
    def owner_nodes(self, value):
        self.__owner_nodes.append(value)

    @property
    def owner_node_count(self):
        return len(self.__owner_nodes)

    def set_var(self, key: str, var: Any):
        """
        Add extra var
        """
        self.set_object(key, var)
        return

    def get_var(self, key: str):
        self.get_object(key)
        return

    def __init_vars(self):
        self.set_object("data_loader", None)  # Data loader
        self.data_loader_collate_fn = None
        self.data_loader_transform = None

        self.set_object("vocab", None)
        self.set_object("vocab_size", None)
        self.set_object("tokenizer", None)
        self.set_object("tokenizer_builder", None)

        self.set_object("data_distribution", None)  # Data distribution
        self.set_object("data_handler", None)  # data_handler

        self.set_object("model", None)  # Model
        self.set_object("model_weight", None)  # model weight
        self.set_object("global_model_weight", None)  # global weight

        self.set_object("optimizer", None)  # optimizer
        self.set_object("loss_func", None)  # loss_func
        self.set_object("training", None)  # training

        self.set_object("aggregation", None)  # aggregation
        self.set_object("client_selection", None)  # Client selection

        self.set_object("strategy", None)  # strategy
        self.set_object("extractor", None)  # extractor

        self.set_object("training_logger", None)  # Training logger

        # Persistent optimizer state: saved across rounds so that momentum /
        # adaptive-learning-rate statistics accumulate correctly.
        # Keyed per model role so client-front and server optimizers stay separate.
        self.set_object("persistent_optimizer_state", None)
        return

    # Properties
    #region
    #---------------------------------------------------------------
    # training_logger property
    @property
    def training_logger(self) -> TrainingLogger:
        return self.get_object("training_logger")

    @training_logger.setter
    def training_logger(self, value):
        self.set_object("training_logger", value)

    # data loader property
    @property
    def data_loader(self) -> DatasetLoader:
        return self.get_object("data_loader")

    @data_loader.setter
    def data_loader(self, value):
        self.set_object("data_loader", value)
        # If it's an NLP task and we already have a shared vocab, inject it into the data_loader
        if value is not None and self.vocab is not None:
            if hasattr(value, "vocab") and value.vocab is None:
                value.vocab = self.vocab

    @property
    def model(self) -> nn.Module :
        return self.get_object("model", cast_type=nn.Module)

    @model.setter
    def model(self, value):
        self.set_object("model", value)

    @property
    def model_weight(self):
        return self.get_object("model_weight")

    @model_weight.setter
    def model_weight(self, value):
        self.set_object("model_weight", value)

    @property
    def global_model_weight(self):
        return self.get_object("global_model_weight")

    @global_model_weight.setter
    def global_model_weight(self, value):
        self.set_object("global_model_weight", value)

    @property
    def aggregation(self):
        return self.get_object("aggregation")

    @aggregation.setter
    def aggregation(self, value):
        self.set_object("aggregation", value)

    @property
    def client_selection(self) -> FedClientSelector:
        return self.get_object("client_selection")

    @client_selection.setter
    def client_selection(self, value):
        self.set_object("client_selection", value)

    @property
    def data_distribution(self) -> DataDistribution:
        return self.get_object("data_distribution")

    @data_distribution.setter
    def data_distribution(self, value):
        self.set_object("data_distribution", value)

    @property
    def tokenizer(self):
        return self.get_object("tokenizer")

    @tokenizer.setter
    def tokenizer(self, value):
        self.set_object("tokenizer", value)

    @property
    def tokenizer_builder(self):
        return self.get_object("tokenizer_builder")

    @tokenizer_builder.setter
    def tokenizer_builder(self, value):
        self.set_object("tokenizer_builder", value)

    @property
    def vocab(self):
        return self.get_object("vocab")

    @vocab.setter
    def vocab(self, value):
        self.set_object("vocab", value)

    @property
    def vocab_size(self):
        return self.get_object("vocab_size")

    @vocab_size.setter
    def vocab_size(self, value):
        self.set_object("vocab_size", value)

    @property
    def loss_func(self):
        return self.get_object("loss_func")

    @loss_func.setter
    def loss_func(self, value):
        self.set_object("loss_func", value)

    @property
    def optimizer(self):
        return self.get_object("optimizer")

    @optimizer.setter
    def optimizer(self, value):
        self.set_object("optimizer", value)

    @property
    def persistent_optimizer_state(self) -> dict | None:
        """Optimizer state_dict saved at the end of each training round.
        Loaded at the start of the next round so momentum / adaptive-LR
        statistics accumulate across rounds per client."""
        return self.get_object("persistent_optimizer_state")

    @persistent_optimizer_state.setter
    def persistent_optimizer_state(self, value: dict | None):
        self.set_object("persistent_optimizer_state", value)

    @property
    def training(self):
        return self.get_object("training")

    @training.setter
    def training(self, value):
        self.set_object("training", value)

    @property
    def strategy(self):
        return self.get_object("strategy")

    @strategy.setter
    def strategy(self, value):
        self.set_object("strategy", value)

    @property
    def extractor(self):
        return self.get_object("extractor")

    @extractor.setter
    def extractor(self, value):
        self.set_object("extractor", value)

    #endregion

    # Prepare variables
    # region
    def prepare_data_loader(self):
        if "data_loader" in self.config_dict:
            data_loader_args = DatasetLoaderArgs(self.config_dict["data_loader"])
            data_loader_args.transform = self.data_loader_transform
            if data_loader_args.task_type == "nlp":
                data_loader_args.tokenizer = self.tokenizer
                data_loader_args.vocab = FedNodeVars.share_vocab
                # Propagate max_len from tokenizer YAML to data loader
                if "tokenizer" in self.config_dict:
                    data_loader_args.max_len = self.config_dict["tokenizer"].get("max_len", None)

            # Assign a per-node generator seed so each client's DataLoader shuffle
            # sequence is independent of the global torch RNG and of other clients.
            # node_id may be an int or a string like "client_0"; we derive a stable
            # integer from it and offset from the global seed (42).
            node_id = getattr(self, "node_id", None)
            if node_id is not None:
                try:
                    id_int = int(node_id)
                except (TypeError, ValueError):
                    id_int = hash(str(node_id)) & 0xFFFF
                data_loader_args.generator_seed = 42 + id_int

            self.data_loader = DatasetLoaderFactory.create(data_loader_args)
            data_loader_args.is_train = False  # get test data loader
            self.test_data_loader = DatasetLoaderFactory.create(data_loader_args)
            self.data_sample_num = self.data_loader.data_sample_num
        
            if data_loader_args.task_type == "nlp" and data_loader_args.vocab_size != None:
                self.vocab_size = data_loader_args.vocab_size
        # Raise event
        args = FedNodeEventArgs("data_loader", self.config_dict).with_sender(self).with_data(self.data_loader)
        self.raise_event("on_prepare_data_loader", args)
        return

    def prepare_data_distribution(self):
        if "data_distribution" in self.config_dict:
            DataDistribution.parse_config(self.config_dict["data_distribution"])
        self.data_distribution = DataDistribution.get()

        args = FedNodeEventArgs("data_distribution", self.config_dict).with_sender(self).with_data(self.data_distribution)
        self.raise_event("on_prepare_data_distribution", args)
        return

    def prepare_data_handler(self):
        args = FedNodeEventArgs("data_handler", self.config_dict).with_sender(self)
        self.raise_event("on_prepare_data_handler", args)
        return

    def prepare_model(self):
        # create model
        if "nn_model" in self.config_dict:
            config = self.config_dict["nn_model"]
        else:
            config = self.config_dict

        name = config.get("name")

        if String.is_none_or_empty(name):
            raise ValueError("Error: Missing model type in yaml.")
        
        if self.config_dict["data_loader"]["task_type"] != "nlp":
            is_share_model = config.get("share_model", True)  # NOTICE: Share model
            if is_share_model and FedNodeVars.share_model is not None:
                self.model = FedNodeVars.share_model
                self.model_weight = ModelUtils.unwrap_model(self.model).state_dict()  # model weight
                console.debug(f"[prepare_model] Reusing shared model: id={id(self.model)}")
            else:
                args = NNModelFactory.create_args(config)
                self.model = NNModelFactory.create(args)
                self.model_weight = ModelUtils.unwrap_model(self.model).state_dict()  # model weight
                console.debug(f"[prepare_model] Created new model: id={id(self.model)}")

            if is_share_model and FedNodeVars.share_model is None:
                FedNodeVars.share_model = self.model
                console.debug(f"[prepare_model] Set share_model: id={id(self.model)}")

        elif self.config_dict["data_loader"]["task_type"] == "nlp":
            is_share_model = config.get("share_model", True)  # NOTICE: Share model
            if is_share_model and FedNodeVars.share_model is not None:
                self.model = FedNodeVars.share_model
                self.model_weight = ModelUtils.unwrap_model(self.model).state_dict()  # model weight
                console.debug(f"[prepare_model] Reusing shared model (NLP): id={id(self.model)}")
            else:
                args = NNModelFactory.create_args(config)
                args.vocab_size = self.vocab_size
                self.model = NNModelFactory.create(args)
                self.model_weight = ModelUtils.unwrap_model(self.model).state_dict()  # model weight
                console.debug(f"[prepare_model] Created new model (NLP): id={id(self.model)}")

            if is_share_model and FedNodeVars.share_model is None:
                FedNodeVars.share_model = self.model
                console.debug(f"[prepare_model] Set share_model (NLP): id={id(self.model)}")

        # Raise event
        args = FedNodeEventArgs("model", self.config_dict).with_sender(self).with_data(self.model)
        self.raise_event("on_prepare_model", args)
        return

    def prepare_optimizer(self):
        # build optimizer
        if "optimizer" in self.config_dict:
            self.optimizer_builder = OptimizerBuilder(self.model.parameters(), self.config_dict)
            self.optimizer = self.optimizer_builder.build()

        args = FedNodeEventArgs("optimizer", self.config_dict).with_sender(self).with_data(self.optimizer)
        self.raise_event("on_prepare_optimizer", args)
        return

    def prepare_loss_func(self):
        # build loss function
        if "loss_func" in self.config_dict:
            self.loss_func = LossFunctionBuilder.build(self.config_dict["loss_func"])

        args = FedNodeEventArgs("loss_func", self.config_dict).with_sender(self).with_data(self.loss_func)
        self.raise_event("on_prepare_loss_func", args)
        return

    def prepare_client_selection(self):
        if "client_selection" in self.config_dict:
            client_selection_args = FedClientSelectorArgs(self.config_dict["client_selection"])
            self.client_selection = FedClientSelectorFactory.create(client_selection_args)

        args = FedNodeEventArgs("client_selection", self.config_dict).with_sender(self).with_data(self.client_selection)
        self.raise_event("on_prepare_client_selection", args)
        return

    def prepare_trainer(self):
        args = FedNodeEventArgs("training", self.config_dict).with_sender(self)

        # build trainer
        trainer_args = ModelTrainerFactory.create_args(self.config_dict["trainer"])
        trainer_args.device = self.device
        trainer_type = self.config_dict.get("trainer", {}).get("trainer_type", trainer_args.trainer_type)
        trainer_args.set_trainer_args(self.model, self.optimizer, self.loss_func, self.data_loader, trainer_type)
        self.trainer = ModelTrainerFactory.create(trainer_args)

        # The trainer constructor eagerly moves the model to the GPU. With one
        # node per client (+ server), leaving every prepared model resident on
        # the GPU would consume N * model_size of VRAM before training even
        # starts. Park the prepared model back on CPU; local_training_step()
        # deep-copies it and moves only the active working copy to the GPU,
        # so peak VRAM stays at ~1 model regardless of client count.
        try:
            self.model = self.model.to("cpu")
            self.trainer.trainer_args.model = self.model
            if hasattr(self.trainer, "model"):
                self.trainer.model = self.model
            ModelUtils.clear_cuda_cache()
        except Exception as e:
            console.warn(f"[prepare_trainer] Could not park model on CPU: {e}")

        self.raise_event("on_prepare_trainer", args)
        return

    def prepare_aggregation(self):
        # Raise strategy event
        args = FedNodeEventArgs("model_aggregation", self.config_dict).with_sender(self)

        #########
        if "aggregation" in self.config_dict:
            aggregation_config = dict(self.config_dict["aggregation"])
            # Canonicalization is a server pipeline option, kept independent of
            # the aggregation method section. Nested configuration remains
            # supported for direct aggregator construction and old config tools.
            if "canonicalization" in self.config_dict and "canonicalization" not in aggregation_config:
                aggregation_config["canonicalization"] = self.config_dict["canonicalization"]
            self.aggregation_method = FedAggregatorFactory.create_aggregator(FedAggregatorArgs(aggregation_config))

        self.raise_event("on_prepare_aggregation", args)
        return
    
    def prepare_model_evaluator(self):
        # Detect task name for NLP/GLUE metric auto-selection
        task_name = None
        dl_cfg = self.config_dict.get("data_loader", {})
        if dl_cfg.get("task_type") == "nlp":
            task_name = dl_cfg.get("name", None)

        trainer_cfg = self.config_dict.get("trainer", {})
        self.model_evaluator = ModelEvaluator(
            self.model, self.data_loader, self.loss_func, self.device,
            task_name=task_name,
            legacy_eval_rng_compat=bool(trainer_cfg.get("legacy_eval_rng_compat", False)),
        )
        return

    def prepare_strategy(self):
        if "strategy" in self.config_dict:
            self.strategy = self.config_dict["strategy"]
        # Raise strategy event
        args = FedNodeEventArgs("strategy", self.config_dict).with_sender(self)
        self.strategy = StrategyFactory.create(StrategyFactory.create_args(self.config_dict["strategy"]), self.owner_nodes[0])
        self.raise_event("on_prepare_strategy", args)
        return

    def prepare_extractor(self):
        # Raise extractor event
        args = FedNodeEventArgs("extractor", self.config_dict).with_sender(self)

        #########
        console.error("TODO: prepare_extractor...")

        self.raise_event("on_prepare_extractor", args)

    def prepare_training_logger(self):
        if "training_logger" in self.config_dict:
            # Auto-configure prefix from experiment config filename (set by AppEntry.load_app_config)
            from ..ml_utils.app_entry import AppEntry
            exp_name = getattr(AppEntry, '_experiment_name', '')
            if exp_name and not self.config_dict["training_logger"].get("prefix"):
                self.config_dict["training_logger"]["prefix"] = f"{exp_name}_"
            self.training_logger = TrainingLogger(self.config_dict["training_logger"])

        # Raise event
        args = FedNodeEventArgs("training_logger", self.config_dict).with_sender(self).with_data(self.training_logger)
        self.raise_event("on_prepare_training_logger", args)
        return

    def prepare_global_inference_model(self):
        if "rank_distribution" in self.config_dict:
            from ..ml_algorithms.lora.lora_utils import LoRAUtils

            config = self.config_dict["nn_model"]
            config['rank_ratio'] = max(self.config_dict["rank_distribution"]["rank_ratio_list"])
            args = NNModelFactory.create_args(config)

            if self.config_dict["data_loader"]["task_type"] != "nlp":
                self.inference_model = NNModelFactory.create(args)
                aligned_weight = LoRAUtils.replace_weight_and_bias(self.inference_model.state_dict(), ModelUtils.unwrap_model(self.model).state_dict())
                self.model_evaluator.change_model(self.inference_model, aligned_weight)

            elif self.config_dict["data_loader"]["task_type"] == "nlp":
                args = NNModelFactory.create_args(config)
                args.vocab_size = self.vocab_size
                self.inference_model = NNModelFactory.create(args)
                aligned_weight = LoRAUtils.replace_weight_and_bias(self.inference_model.state_dict(), ModelUtils.unwrap_model(self.model).state_dict())
                self.model_evaluator.change_model(self.inference_model, aligned_weight)

            # ── Keep inference model on CPU between evaluations to save GPU memory ──
            console.debug(
                f"[prepare_global_inference_model] inference_model id={id(self.inference_model)}, "
                f"share_model id={id(FedNodeVars.share_model)}"
            )

            args = FedNodeEventArgs("lora_inference_model", self.config_dict).with_sender(self)
            self.raise_event("on_prepare_lora_inference_model", args)
            
        return

    def prepare_vocab_tokenizer(self):
        if "tokenizer" in self.config_dict and self.config_dict["data_loader"]["task_type"] == "nlp":
            use_hf = bool(self.config_dict.get("tokenizer", {}).get("use_hf_tokenizer", False))

            # Build tokenizer (torchtext or HF). TokenizerBuilder.meta holds hf_tokenizer when available.
            self.tokenizer_builder = TokenizerBuilder(self.config_dict)
            built = self.tokenizer_builder.build()

            if use_hf and "hf_tokenizer" in getattr(self.tokenizer_builder, "meta", {}):
                console.info("Using HuggingFace tokenizer...")
                self.tokenizer = self.tokenizer_builder.meta.get("hf_tokenizer")
                # Prefer hf vocab size if present
                self.vocab_size = getattr(self.tokenizer, "vocab_size", None)
            else:
                # legacy torchtext tokenizer function
                self.tokenizer = built

            args = FedNodeEventArgs("tokenizer", self.config_dict).with_sender(self).with_data(self.tokenizer)
            self.raise_event("on_prepare_tokenizer", args)

        return

    def prepare_vocab(self):
        # Determine task type
        task_type = "image"
        if self.data_loader is not None:
            task_type = self.data_loader.task_type
        elif "data_loader" in self.config_dict:
            task_type = self.config_dict["data_loader"].get("task_type", "image")

        if task_type == "nlp":
            use_hf = bool(self.config_dict.get("tokenizer", {}).get("use_hf_tokenizer", False))
            if use_hf:
                console.info("Using HuggingFace tokenizer vocab...")
                # HF tokenizer path: skip vocab building, rely on tokenizer.vocab_size
                if self.vocab_size is None and self.tokenizer is not None:
                    self.vocab_size = getattr(self.tokenizer, "vocab_size", None)
                self.vocab = None
            else:
                if FedNodeVars.share_vocab is None and self.data_loader is not None:
                    if hasattr(self.data_loader, "vocab") and self.data_loader.vocab is not None:
                        FedNodeVars.share_vocab = self.data_loader.vocab
                    else:
                        console.info("Building global vocab from server data...")
                        data_input = self.data_loader.data_set
                        FedNodeVars.share_vocab = TokenizerBuilder.build_vocab(data_input, self.tokenizer)
                
                self.vocab = FedNodeVars.share_vocab
                if self.vocab is not None:
                    self.vocab_size = len(self.vocab)
                    if self.data_loader is not None and hasattr(self.data_loader, "vocab"):
                        self.data_loader.vocab = self.vocab

        # Raise event
        args = FedNodeEventArgs("vocab", self.config_dict).with_sender(self).with_data(self.vocab)
        self.raise_event("on_prepare_vocab", args)
        return

    def prepare(self) -> Any:
        """
        Prepare components. In a simulated environment, clients skip redundant data loading
        and server-only configurations.
        """
        # Determine if this node acts as a server based on its configuration (id or prefix)
        role = self.config_dict.get("role", "client")

        console.info(f"========================== [ Prepare {role.upper()} ] ==========================")

        # ── Log train optimization status (once per process, on server) ──
        if role == "server":
            self._log_optimization_status()

        if role == "server":
            self._prepare_role_server()
        else:
            self._prepare_role_client()

        console.info(f"{role} prepare completed.")
        console.ok(f"========================== [ {role.upper()} READY ] ==========================")

        return self

    def _prepare_role_server(self):
        """Prepare steps specific to Server node"""
        console.info("Prepare vocab tokenizer (Server)...", "")
        self.prepare_vocab_tokenizer()
        console.ok("OK")

        console.info("Prepare data loader (Server)...", "")
        self.prepare_data_loader()
        self.prepare_vocab()
        console.ok("OK")

        console.info("Prepare data_distribution...", "")
        self.prepare_data_distribution()
        console.ok("OK")

        console.info("Prepare data handler...", "")
        self.prepare_data_handler()
        console.ok("OK")

        console.info("Prepare client selection...", "")
        self.prepare_client_selection()
        console.ok("OK")

        console.info(f"Prepare aggregation...", "")
        self.prepare_aggregation()
        console.ok("OK")

        console.info("Prepare NN model...", "")
        self.prepare_model()
        console.ok("OK")

        console.info("Prepare optimizer...", "")
        self.prepare_optimizer()
        console.ok("OK")

        console.info("Prepare loss function...", "")
        self.prepare_loss_func()
        console.ok("OK")

        # Prepare logger
        console.info("Prepare training logger...", "")
        self.prepare_training_logger()
        console.ok("OK")

        console.info("Prepare model evaluator...", "")
        self.prepare_model_evaluator()
        console.ok("OK")
        
        console.info(f"Prepare trainer...", "")
        self.prepare_trainer()
        console.ok("OK")

        console.info("check global model for inference", "")
        self.prepare_global_inference_model()
        console.ok("OK")
        return

    def _prepare_role_client(self):
        """Prepare steps specific to Client node"""
        # For clients, we may still need vocab/tokenizer information if it's an NLP task

        console.info("Prepare vocab tokenizer (Client)...", "")
        self.prepare_vocab_tokenizer()
        console.ok("OK")

        console.info("Prepare vocab (Client)...", "")
        self.prepare_vocab()
        console.ok("OK")

        console.info("Prepare NN model...", "")
        self.prepare_model()
        console.ok("OK")

        console.info("Prepare optimizer...", "")
        self.prepare_optimizer()
        console.ok("OK")

        console.info("Prepare loss function...", "")
        self.prepare_loss_func()
        console.ok("OK")
        
        console.info(f"Prepare trainer...", "")
        self.prepare_trainer()
        console.ok("OK")
        return

    def prepare_strategy_only(self):
        console.info("Prepare strategy...", "")
        self.prepare_strategy()
        console.ok("OK")
        return self
    
    def set_device(self, device: str):
        """
        Set computation device
        """
        self.device = device
        trainer = getattr(self, "trainer", None)
        if trainer is not None and hasattr(trainer, "trainer_args"):
            trainer.trainer_args.device = device
        return

    # ------------------------------------------------------------------
    # Train optimization status logging
    # ------------------------------------------------------------------
    @staticmethod
    def _log_optimization_status() -> None:
        """Log which train optimizations are enabled and their bit-exact impact."""
        from ..ml_utils.training_utils import TrainingUtils
        status = TrainingUtils.get_optimization_status()
        if not status:
            return

        console.info("── Train Optimization Config ──")
        for key, entry in status.items():
            enabled = entry.get("enabled", False)
            tag = "ON " if enabled else "OFF"
            line = f"  [{tag}] {key}"
            warning = entry.get("warning")
            if warning:
                line += f"  {warning}"
            if enabled:
                console.ok(line) if not warning else console.warn(line)
            else:
                console.debug(line)
        console.info("────────────────────────────────")
