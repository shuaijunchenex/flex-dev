from __future__ import annotations
from .model_trainer import ModelTrainer, ModelTrainerArgs


class ModelTrainerFactory:
    """
    Model trainer factory
    """
    
    @staticmethod
    def create_args(config_dict: dict, is_clone_dict: bool = False) -> ModelTrainerArgs:
        """
        Static method to create trainer args
        """
        return ModelTrainerArgs(config_dict, is_clone_dict)

    @staticmethod
    def create(args: ModelTrainerArgs) -> ModelTrainer:
        """
        Static method to create trainer.
        Applies ``torch.compile`` to the model when the global
        ``train_optimization.torch_compile`` flag is enabled and
        PyTorch ≥ 2.0 is available.
        """
        trainer = ModelTrainerFactory._create_trainer(args)

        # ── torch.compile (PyTorch 2.0+) ──────────────────────────────
        from ..ml_utils.training_utils import TrainingUtils
        if TrainingUtils.is_optimization_enabled("torch_compile"):
            if hasattr(torch, "compile") and trainer.trainer_args.model is not None:
                try:
                    trainer.trainer_args.model = torch.compile(
                        trainer.trainer_args.model, mode="reduce-overhead"
                    )
                    trainer.model = trainer.trainer_args.model
                    console.debug("[torch.compile] Model compiled successfully.")
                except Exception as e:
                    console.warn(f"[torch.compile] Failed: {e} — running without compilation.")
            else:
                console.debug("[torch.compile] torch.compile not available — skipped.")
        return trainer

    @staticmethod
    def _create_trainer(args: ModelTrainerArgs) -> ModelTrainer:
        match args.trainer_type:
            case "standard":
                from .trainer._model_trainer_standard import ModelTrainer_Standard
                return ModelTrainer_Standard(args)
            case "standard_legacy":
                from .trainer._model_trainer_standard_legacy import ModelTrainer_StandardLegacy
                return ModelTrainer_StandardLegacy(args)
            case "standard_legacy_prepass":
                from .trainer._model_trainer_standard_legacy_prepass import ModelTrainer_StandardLegacyPrepass
                return ModelTrainer_StandardLegacyPrepass(args)
            case "vit":
                from .trainer._model_trainer_vit import ModelTrainer_Vit
                return ModelTrainer_Vit(args)
            case "imdb":
                from .trainer._model_trainer_imdb import ModelTrainer_Imdb
                return ModelTrainer_Imdb(args)
            case "glue":
                from .trainer._model_trainer_glue import ModelTrainer_GLUE
                return ModelTrainer_GLUE(args)
            case "complex_cv":
                from .trainer._model_trainer_complex_cv import ModelTrainer_ComplexCV
                return ModelTrainer_ComplexCV(args)
            case "sl_client":
                from .trainer._model_trainer_sl_client import ModelTrainer_SlClient
                return ModelTrainer_SlClient(args)
            case "sl_server":
                from .trainer._model_trainer_sl_server import ModelTrainer_SlServer
                return ModelTrainer_SlServer(args)
            case "pipeline_test":
                from .trainer._model_trainer_pipeline_test import ModelTrainer_PipelineTest
                return ModelTrainer_PipelineTest(args)
            case "glue_test":
                from .trainer._model_trainer_glue_test import ModelTrainer_GlueTest
                return ModelTrainer_GlueTest(args)
            case "lora_grad":
                from .trainer._model_trainer_lora_grad import ModelTrainer_LoraGrad
                return ModelTrainer_LoraGrad(args)
            case "rblasa":
                from .trainer._model_trainer_rblasa import ModelTrainer_RBLASA
                return ModelTrainer_RBLASA(args)
            case "sara":
                from .trainer._model_trainer_sara import ModelTrainer_SARA
                return ModelTrainer_SARA(args)
            case "adaptive_sara":
                from .trainer._model_trainer_adaptive_sara import ModelTrainer_AdaptiveSARA
                return ModelTrainer_AdaptiveSARA(args)
            case "rbla_strong_a":
                from .trainer.rbla_problem._model_trainer_rbla_strong_a import ModelTrainer_RBLAStrongA
                return ModelTrainer_RBLAStrongA(args)
            case "fedgra":
                from .trainer._model_trainer_fedgra import ModelTrainer_FedGRA
                return ModelTrainer_FedGRA(args)
            case _:
                raise Exception(f"Undefined trainer type {args.trainer_type}.")
