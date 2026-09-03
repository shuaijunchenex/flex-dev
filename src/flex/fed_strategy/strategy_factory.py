from __future__ import annotations
from typing import Callable

from .strategy_args import StrategyArgs
from .client_strategy import ClientStrategy
from .server_strategy import ServerStrategy

class StrategyFactory:
    """
    " Dataset loader factory
    """

    @staticmethod
    def create_args(config_dict: dict, is_clone_dict: bool = False) -> StrategyArgs:
        """
        " Static method to create data loader args
        """
        return StrategyArgs(config_dict, is_clone_dict)

    @staticmethod
    def create_runner_args(config_dict: dict, is_clone_dict: bool = False) -> StrategyArgs:
        """
        " Static method to create runner strategy args
        """
        runner_config = {**config_dict["general"], **config_dict["strategy"]}
        return StrategyArgs(runner_config, is_clone_dict)

    @staticmethod
    def create(args: StrategyArgs, node):
        match args.role.lower():
            case "client":
                return StrategyFactory.create_client_strategy(args, node)
            case "server":
                return StrategyFactory.create_server_strategy(args, node)

    @staticmethod
    def create_runner_strategy(runner_strategy_args: StrategyArgs, runner, client_nodes, server_node) -> ClientStrategy:
        """
        " Static method to create runner strategy
        """
        match runner_strategy_args.strategy_name.lower():
            case "fedavg":
                # Import FedAvgRunnerStrategy from the appropriate module
                from flex.fed_strategy.runner_strategy_impl._fedavg_runner_strategy import FedAvgRunnerStrategy
                return FedAvgRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "observer":
                from flex.fed_strategy.runner_strategy_impl._observer_runner_strategy import ObserverRunnerStrategy
                return ObserverRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "oort":
                from flex.fed_strategy.runner_strategy_impl._oort_runner_strategy import OortRunnerStrategy
                return OortRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "fedgra" | "fedgra_standard" | "fedgra_hpc":
                from flex.fed_strategy.runner_strategy_impl._fedgra_runner_strategy import FedgraRunnerStrategy
                return FedgraRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "fedgra_keras_exact":
                from flex.fed_strategy.runner_strategy_impl._fedgra_keras_exact_runner_strategy import FedgraKerasExactRunnerStrategy
                return FedgraKerasExactRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "pyramidfl":
                from flex.fed_strategy.runner_strategy_impl._pyramidfl_runner_strategy import PyramidFLRunnerStrategy
                return PyramidFLRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "rbla":
                from flex.fed_strategy.runner_strategy_impl._rbla_runner_strategy import RblaRunnerStrategy
                return RblaRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "sp_plus":
                from flex.fed_strategy.runner_strategy_impl._sp_plus_runner_strategy import SpPlusRunnerStrategy
                return SpPlusRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "rbla_plus_rank_scale":
                from flex.fed_strategy.runner_strategy_impl._rbla_plus_rank_scale_runner_strategy import RblaPlusRankScaleRunnerStrategy
                return RblaPlusRankScaleRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "rbla_refdiag":
                from flex.fed_strategy.runner_strategy_impl.rbla_problem._rbla_reference_runner_strategy import RblaRefDiagRunnerStrategy
                return RblaRefDiagRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "rbla_freeze_a":
                from flex.fed_strategy.runner_strategy_impl.rbla_problem._rbla_reference_runner_strategy import RblaFreezeARunnerStrategy
                return RblaFreezeARunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "rbla_strong_a":
                from flex.fed_strategy.runner_strategy_impl.rbla_problem._rbla_reference_runner_strategy import RblaStrongARunnerStrategy
                return RblaStrongARunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "rbla_refdiag_analysis" | "rbla_freeze_a_analysis" | "rbla_strong_a_analysis" | "rbla_freeze_a_support_gamma" | "rbla_p8_refdiag_support_scaling" | "rbla_p8_strong_a_support_scaling" | "rbla_p9_freeze_a_support_scaling" | "rbla_p10_freeze_a_support_scaling":
                from flex.fed_strategy.runner_strategy_impl.rbla_problem._rbla_support_analysis_runner import RblaSupportAnalysisRunnerStrategy
                return RblaSupportAnalysisRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "rbla_sasg":
                from flex.fed_strategy.runner_strategy_impl._rbla_sasg_runner_strategy import RblaSasgRunnerStrategy
                return RblaSasgRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "sara":
                from flex.fed_strategy.runner_strategy_impl._sara_runner_strategy import SaraRunnerStrategy
                return SaraRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "adaptive_sara":
                from flex.fed_strategy.runner_strategy_impl._adaptive_sara_runner_strategy import AdaptiveSaraRunnerStrategy
                return AdaptiveSaraRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "sp":
                from flex.fed_strategy.runner_strategy_impl._sp_runner_strategy import SpRunnerStrategy
                return SpRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "flora":
                from flex.fed_strategy.runner_strategy_impl._flora_runner_strategy import FloraRunnerStrategy
                return FloraRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "lora_svd":
                from flex.fed_strategy.runner_strategy_impl._lora_svd_analysis_runner_strategy import LoraSvdAnalysisRunnerStrategy
                return LoraSvdAnalysisRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "sfl":
                from flex.sfl_strategy.runner_strategy_impl._sfl_runner_example import SflRunnerStrategy
                return SflRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "sfl_roundavg":
                from flex.sfl_strategy.runner_strategy_impl._sfl_roundavg_runner import SflRoundAvgRunnerStrategy
                return SflRoundAvgRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "sfl_aligned":
                from flex.sfl_strategy.runner_strategy_impl._sfl_aligned_runner import SflAlignedRunnerStrategy
                return SflAlignedRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)
            case "rblasa":
                from flex.fed_strategy.runner_strategy_impl._rblasa_runner_strategy import RblasaRunnerStrategy
                return RblasaRunnerStrategy(runner, runner_strategy_args, client_nodes, server_node)

        raise ValueError(f"Runner strategy type '{runner_strategy_args.strategy_name}' not support.")

    @staticmethod
    def create_client_strategy(client_strategy_args: StrategyArgs, client_node_input) -> ClientStrategy:
        """
        " Static method to create data loader
        """
        match client_strategy_args.strategy_name.lower():
            case "fedavg":
                from flex.fed_strategy.client_strategy_impl._fedavg_client import FedAvgClientTrainingStrategy
                return FedAvgClientTrainingStrategy(client_strategy_args, client_node_input)
            case "client_selection_purpose":
                from flex.fed_strategy.client_strategy_impl._client_selection_purpose_client import ClientSelectionPurposeClientStrategy
                return ClientSelectionPurposeClientStrategy(client_strategy_args, client_node_input)
            case "oort":
                from flex.fed_strategy.client_strategy_impl._oort_client import OortClientTrainingStrategy
                return OortClientTrainingStrategy(client_strategy_args, client_node_input)
            case "pyramidfl":
                from flex.fed_strategy.client_strategy_impl._pyramidfl_client import PyramidFLClientTrainingStrategy
                return PyramidFLClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rbla":
                from flex.fed_strategy.client_strategy_impl._rbla_client import RblaClientTrainingStrategy
                return RblaClientTrainingStrategy(client_strategy_args, client_node_input)
            case "sp_plus":
                from flex.fed_strategy.client_strategy_impl._sp_plus_client import SpPlusClientTrainingStrategy
                return SpPlusClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rbla_plus_rank_scale":
                from flex.fed_strategy.client_strategy_impl._rbla_plus_rank_scale_client import RblaPlusRankScaleClientTrainingStrategy
                return RblaPlusRankScaleClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rbla_refdiag":
                from flex.fed_strategy.client_strategy_impl.rbla_problem._rbla_reference_client import RblaRefDiagClientTrainingStrategy
                return RblaRefDiagClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rbla_freeze_a":
                from flex.fed_strategy.client_strategy_impl.rbla_problem._rbla_reference_client import RblaFreezeAClientTrainingStrategy
                return RblaFreezeAClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rbla_strong_a":
                from flex.fed_strategy.client_strategy_impl.rbla_problem._rbla_reference_client import RblaStrongAClientTrainingStrategy
                return RblaStrongAClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rbla_refdiag_analysis":
                from flex.fed_strategy.client_strategy_impl.rbla_problem._rbla_reference_client import RblaRefDiagAnalysisClientTrainingStrategy
                return RblaRefDiagAnalysisClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rbla_freeze_a_analysis":
                from flex.fed_strategy.client_strategy_impl.rbla_problem._rbla_reference_client import RblaFreezeAAnalysisClientTrainingStrategy
                return RblaFreezeAAnalysisClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rbla_strong_a_analysis":
                from flex.fed_strategy.client_strategy_impl.rbla_problem._rbla_reference_client import RblaStrongAAnalysisClientTrainingStrategy
                return RblaStrongAAnalysisClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rbla_freeze_a_support_gamma":
                from flex.fed_strategy.client_strategy_impl.rbla_problem._rbla_reference_client import RblaFreezeASupportGammaClientTrainingStrategy
                return RblaFreezeASupportGammaClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rbla_p8_refdiag_support_scaling":
                from flex.fed_strategy.client_strategy_impl.rbla_problem._rbla_reference_client import RblaP8RefDiagSupportScalingClientTrainingStrategy
                return RblaP8RefDiagSupportScalingClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rbla_p8_strong_a_support_scaling":
                from flex.fed_strategy.client_strategy_impl.rbla_problem._rbla_reference_client import RblaP8StrongASupportScalingClientTrainingStrategy
                return RblaP8StrongASupportScalingClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rbla_p10_freeze_a_support_scaling":
                from flex.fed_strategy.client_strategy_impl.rbla_problem._rbla_reference_client import RblaP10FreezeASupportScalingClientTrainingStrategy
                return RblaP10FreezeASupportScalingClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rbla_p9_freeze_a_support_scaling":
                from flex.fed_strategy.client_strategy_impl.rbla_problem._rbla_reference_client import RblaP9FreezeASupportScalingClientTrainingStrategy
                return RblaP9FreezeASupportScalingClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rbla_sasg":
                from flex.fed_strategy.client_strategy_impl._rbla_sasg_client import RblaSasgClientTrainingStrategy
                return RblaSasgClientTrainingStrategy(client_strategy_args, client_node_input)
            case "sara":
                from flex.fed_strategy.client_strategy_impl._sara_client import SaraClientTrainingStrategy
                return SaraClientTrainingStrategy(client_strategy_args, client_node_input)
            case "adaptive_sara":
                from flex.fed_strategy.client_strategy_impl._adaptive_sara_client import AdaptiveSaraClientTrainingStrategy
                return AdaptiveSaraClientTrainingStrategy(client_strategy_args, client_node_input)
            case "sp":
                from flex.fed_strategy.client_strategy_impl._sp_client import SpClientTrainingStrategy
                return SpClientTrainingStrategy(client_strategy_args, client_node_input)
            case "flora":
                from flex.fed_strategy.client_strategy_impl._flora_client import FloraClientTrainingStrategy
                return FloraClientTrainingStrategy(client_strategy_args, client_node_input)
            case "rblasa":
                from flex.fed_strategy.client_strategy_impl._rblasa_client import RblasaClientTrainingStrategy
                return RblasaClientTrainingStrategy(client_strategy_args, client_node_input)
            case "sfl":
                from flex.sfl_strategy.client_strategy_impl._sfl_client_example import SflClientStrategy
                return SflClientStrategy(client_strategy_args, client_node_input)
            case "sfl_roundavg":
                from flex.sfl_strategy.client_strategy_impl._sfl_roundavg_client import SflRoundAvgClientStrategy
                return SflRoundAvgClientStrategy(client_strategy_args, client_node_input)
            case "sfl_aligned":
                from flex.sfl_strategy.client_strategy_impl._sfl_aligned_client import SflAlignedClientStrategy
                return SflAlignedClientStrategy(client_strategy_args, client_node_input)

        raise ValueError(f"Client strategy type '{client_strategy_args.strategy_name}' not support.")

    @staticmethod
    def create_server_strategy(server_strategy_args: StrategyArgs, serve_node_input) -> ServerStrategy:
        """
        " Static method to create server strategy
        """
        match server_strategy_args.strategy_name.lower():
            case "fedavg":
                from flex.fed_strategy.server_strategy_impl._fedavg_server import FedAvgServerStrategy
                return FedAvgServerStrategy(server_strategy_args, serve_node_input)
            case "oort":
                from flex.fed_strategy.server_strategy_impl._oort_server import OortServerStrategy
                return OortServerStrategy(server_strategy_args, serve_node_input)
            case "adafl":
                from flex.fed_strategy.server_strategy_impl._adafl_server import AdaFLServerStrategy
                return AdaFLServerStrategy(server_strategy_args, serve_node_input)
            case "fedgra" | "fedgra_standard" | "fedgra_hpc":
                from flex.fed_strategy.server_strategy_impl._fedgra_server import FedgraServerStrategy
                return FedgraServerStrategy(server_strategy_args, serve_node_input)
            case "pyramidfl":
                from flex.fed_strategy.server_strategy_impl._pyramidfl_server import PyramidFLServerStrategy
                return PyramidFLServerStrategy(server_strategy_args, serve_node_input)
            case "fedsdr":
                from flex.fed_strategy.server_strategy_impl._fedsdr_server import FedSDRServerStrategy
                return FedSDRServerStrategy(server_strategy_args, serve_node_input)
            case "repufl":
                from flex.fed_strategy.server_strategy_impl._repufl_server import RepuFLServerStrategy
                return RepuFLServerStrategy(server_strategy_args, serve_node_input)
            case "rbla":
                from flex.fed_strategy.server_strategy_impl._rbla_server import RblaServerStrategy
                return RblaServerStrategy(server_strategy_args, serve_node_input)
            case "sp_plus":
                from flex.fed_strategy.server_strategy_impl._sp_plus_server import SpPlusServerStrategy
                return SpPlusServerStrategy(server_strategy_args, serve_node_input)
            case "rbla_plus_rank_scale":
                from flex.fed_strategy.server_strategy_impl._rbla_plus_rank_scale_server import RblaPlusRankScaleServerStrategy
                return RblaPlusRankScaleServerStrategy(server_strategy_args, serve_node_input)
            case "rbla_refdiag":
                from flex.fed_strategy.server_strategy_impl.rbla_problem._rbla_reference_server import RblaRefDiagServerStrategy
                return RblaRefDiagServerStrategy(server_strategy_args, serve_node_input)
            case "rbla_freeze_a":
                from flex.fed_strategy.server_strategy_impl.rbla_problem._rbla_reference_server import RblaFreezeAServerStrategy
                return RblaFreezeAServerStrategy(server_strategy_args, serve_node_input)
            case "rbla_strong_a":
                from flex.fed_strategy.server_strategy_impl.rbla_problem._rbla_reference_server import RblaStrongAServerStrategy
                return RblaStrongAServerStrategy(server_strategy_args, serve_node_input)
            case "rbla_refdiag_analysis":
                from flex.fed_strategy.server_strategy_impl.rbla_problem._rbla_support_analysis_server import RblaRefDiagAnalysisServerStrategy
                return RblaRefDiagAnalysisServerStrategy(server_strategy_args, serve_node_input)
            case "rbla_freeze_a_analysis":
                from flex.fed_strategy.server_strategy_impl.rbla_problem._rbla_support_analysis_server import RblaFreezeAAnalysisServerStrategy
                return RblaFreezeAAnalysisServerStrategy(server_strategy_args, serve_node_input)
            case "rbla_strong_a_analysis":
                from flex.fed_strategy.server_strategy_impl.rbla_problem._rbla_support_analysis_server import RblaStrongAAnalysisServerStrategy
                return RblaStrongAAnalysisServerStrategy(server_strategy_args, serve_node_input)
            case "rbla_freeze_a_support_gamma":
                from flex.fed_strategy.server_strategy_impl.rbla_problem._rbla_support_analysis_server import RblaFreezeASupportGammaServerStrategy
                return RblaFreezeASupportGammaServerStrategy(server_strategy_args, serve_node_input)
            case "rbla_p8_refdiag_support_scaling":
                from flex.fed_strategy.server_strategy_impl.rbla_problem._rbla_support_analysis_server import RblaP8RefDiagSupportScalingServerStrategy
                return RblaP8RefDiagSupportScalingServerStrategy(server_strategy_args, serve_node_input)
            case "rbla_p8_strong_a_support_scaling":
                from flex.fed_strategy.server_strategy_impl.rbla_problem._rbla_support_analysis_server import RblaP8StrongASupportScalingServerStrategy
                return RblaP8StrongASupportScalingServerStrategy(server_strategy_args, serve_node_input)
            case "rbla_p10_freeze_a_support_scaling":
                from flex.fed_strategy.server_strategy_impl.rbla_problem._rbla_support_analysis_server import RblaP10FreezeASupportScalingServerStrategy
                return RblaP10FreezeASupportScalingServerStrategy(server_strategy_args, serve_node_input)
            case "rbla_p9_freeze_a_support_scaling":
                from flex.fed_strategy.server_strategy_impl.rbla_problem._rbla_support_analysis_server import RblaP9FreezeASupportScalingServerStrategy
                return RblaP9FreezeASupportScalingServerStrategy(server_strategy_args, serve_node_input)
            case "lora_svd":
                from flex.fed_strategy.server_strategy_impl._lora_svd_analysis_server import LoraSvdAnalysisServerStrategy
                return LoraSvdAnalysisServerStrategy(server_strategy_args, serve_node_input)
            case "sp":
                from flex.fed_strategy.server_strategy_impl._sp_server import SpServerStrategy
                return SpServerStrategy(server_strategy_args, serve_node_input)
            case "flora":
                from flex.fed_strategy.server_strategy_impl._flora_server import FloraServerStrategy
                return FloraServerStrategy(server_strategy_args, serve_node_input)
            case "close_optimal":
                from flex.fed_strategy.server_strategy_impl._close_optimal_server import CloseOptimalServerStrategy
                return CloseOptimalServerStrategy(server_strategy_args, serve_node_input)
            case "rbla_sasg":
                from flex.fed_strategy.server_strategy_impl._rbla_sasg_server import RblaSasgServerStrategy
                return RblaSasgServerStrategy(server_strategy_args, serve_node_input)
            case "client_selection_purpose":
                from flex.fed_strategy.server_strategy_impl._client_selection_purpose_server import ClientSelectionPurposeServerStrategy
                return ClientSelectionPurposeServerStrategy(server_strategy_args, serve_node_input)
            case "sfl":
                from flex.sfl_strategy.server_strategy_impl._sfl_server_example import SflServerStrategy
                return SflServerStrategy(server_strategy_args, serve_node_input)
            case "sfl_roundavg":
                from flex.sfl_strategy.server_strategy_impl._sfl_roundavg_server import SflRoundAvgServerStrategy
                return SflRoundAvgServerStrategy(server_strategy_args, serve_node_input)
            case "sfl_aligned":
                from flex.sfl_strategy.server_strategy_impl._sfl_aligned_server import SflAlignedServerStrategy
                return SflAlignedServerStrategy(server_strategy_args, serve_node_input)

        raise ValueError(f"Server strategy type '{server_strategy_args.strategy_name}' not support.")
