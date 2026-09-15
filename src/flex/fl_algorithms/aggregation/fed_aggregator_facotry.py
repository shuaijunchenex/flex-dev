from __future__ import annotations

from .fed_aggregator_abc import AbstractFedAggregator
from .fed_aggregator_args import FedAggregatorArgs
from ...ml_utils import console

class FedAggregatorFactory:
    '''
    ' Fed aggregator factory
    '''

    @staticmethod
    def create_args(config_dict: dict, is_clone_dict: bool = False) -> FedAggregatorArgs:
        """
        " Static method to create fed aggregator args
        """
        return FedAggregatorArgs(config_dict, is_clone_dict)

    @staticmethod
    def create_aggregator(args: FedAggregatorArgs) -> AbstractFedAggregator:
        match args.method:
            case "fedavg":
                from .methods._fed_aggregator_fedavg import FedAggregator_FedAvg
                console.debug("Using FedAvg aggregator")
                return FedAggregator_FedAvg(args)
            case "rbla":
                from .methods._fed_aggregator_rbla import FedAggregator_RBLA
                console.debug("Using RBLA aggregator")
                return FedAggregator_RBLA(args)
            case "sp_plus":
                from .methods._fed_aggregator_sp_plus import FedAggregator_SPPlus
                console.debug("Using SP+ aggregator")
                return FedAggregator_SPPlus(args)
            case "rbla_refdiag":
                from .methods.rbla_problem._fed_aggregator_rbla_reference import FedAggregator_RBLARefDiag
                console.debug("Using isolated RBLA reference-diagnostic aggregator")
                return FedAggregator_RBLARefDiag(args)
            case "rbla_freeze_a":
                from .methods.rbla_problem._fed_aggregator_rbla_reference import FedAggregator_RBLAFreezeA
                console.debug("Using isolated RBLA Freeze-A aggregator")
                return FedAggregator_RBLAFreezeA(args)
            case "rbla_strong_a":
                from .methods.rbla_problem._fed_aggregator_rbla_reference import FedAggregator_RBLAStrongA
                console.debug("Using isolated RBLA Strong-A aggregator")
                return FedAggregator_RBLAStrongA(args)
            case "rbla_refdiag_analysis":
                from .methods.rbla_problem._fed_aggregator_rbla_reference import FedAggregator_RBLARefDiagAnalysis
                console.debug("Using isolated RBLA RefDiag analysis aggregator")
                return FedAggregator_RBLARefDiagAnalysis(args)
            case "rbla_freeze_a_analysis":
                from .methods.rbla_problem._fed_aggregator_rbla_reference import FedAggregator_RBLAFreezeAAnalysis
                console.debug("Using isolated RBLA Freeze-A analysis aggregator")
                return FedAggregator_RBLAFreezeAAnalysis(args)
            case "rbla_strong_a_analysis":
                from .methods.rbla_problem._fed_aggregator_rbla_reference import FedAggregator_RBLAStrongAAnalysis
                console.debug("Using isolated RBLA Strong-A analysis aggregator")
                return FedAggregator_RBLAStrongAAnalysis(args)
            case "rbla_freeze_a_support_gamma":
                from .methods.rbla_problem._fed_aggregator_rbla_support_scaling import FedAggregator_RBLAFreezeASupportGamma
                console.debug("Using isolated RBLA Freeze-A support-gamma aggregator")
                return FedAggregator_RBLAFreezeASupportGamma(args)
            case "rbla_p8_refdiag_support_scaling":
                from .methods.rbla_problem._fed_aggregator_rbla_support_scaling import FedAggregator_RBLARefDiagSupportScaling
                console.debug("Using isolated P8 RefDiag support-scaling aggregator")
                return FedAggregator_RBLARefDiagSupportScaling(args)
            case "rbla_p8_strong_a_support_scaling":
                from .methods.rbla_problem._fed_aggregator_rbla_support_scaling import FedAggregator_RBLAStrongASupportScaling
                console.debug("Using isolated P8 Strong-A support-scaling aggregator")
                return FedAggregator_RBLAStrongASupportScaling(args)
            case "rbla_p10_freeze_a_support_scaling":
                from .methods.rbla_problem._fed_aggregator_rbla_support_scaling import FedAggregator_RBLAP10FreezeASupportScaling
                console.debug("Using isolated P10 Freeze-A support-scaling aggregator")
                return FedAggregator_RBLAP10FreezeASupportScaling(args)
            case "rbla_p9_freeze_a_support_scaling":
                from .methods.rbla_problem._fed_aggregator_rbla_support_scaling import FedAggregator_RBLAP9FreezeASupportScaling
                console.debug("Using isolated P9 Freeze-A support-scaling aggregator")
                return FedAggregator_RBLAP9FreezeASupportScaling(args)
            case "zeropadding":
                from .methods._fed_aggregator_zeropadding import FedAggregator_ZeroPadding
                console.debug("Using ZeroPadding aggregator")
                return FedAggregator_ZeroPadding(args)
            case "replication_padding":
                from .methods._fed_aggregator_replication_padding import (
                    FedAggregator_ReplicationPadding,
                )
                console.debug("Using Replication Padding aggregator")
                return FedAggregator_ReplicationPadding(args)
            case "ffalora":
                from .methods._fed_aggregator_ffalora import FedAggregator_FFALoRA
                console.debug("Using FFA-LoRA aggregator")
                return FedAggregator_FFALoRA(args)
            case "flora":
                from .methods._fed_aggregator_flora import FedAggregator_Flora
                console.debug("Using Flora aggregator")
                return FedAggregator_Flora(args)
            case "fedsalora":
                from .methods._fed_aggregator_fedsalora import FedAggregator_FedSALoRA
                console.debug("Using FedSA-LoRA aggregator")
                return FedAggregator_FedSALoRA(args)
            case "rolora":
                from .methods._fed_aggregator_rolora import FedAggregator_RoLoRA
                console.debug("Using RoLoRA aggregator")
                return FedAggregator_RoLoRA(args)
            case "florg":
                from .methods._fed_aggregator_florg import FedAggregator_FLoRG
                console.debug("Using FLoRG aggregator")
                return FedAggregator_FLoRG(args)
            case "sp":
                from .methods._fed_aggregator_sp import FedAggregator_SP
                console.debug("Using Sum-Product aggregator")
                return FedAggregator_SP(args)
            case "sflavg":
                from .methods._sfl_aggregator_sflavg import SflAggregator_SflAvg
                console.debug("Using SFLAvg gradient aggregator")
                return SflAggregator_SflAvg(args)
            case "close_optimal":
                from .methods._fed_aggregator_close_optimal import FedAggregator_CloseOptimal
                console.debug("Using CloseOptimal aggregator")
                return FedAggregator_CloseOptimal(args)
            case "rbla_sasg":
                from .methods._fed_aggregator_rbla_sasg import FedAggregator_RBLA_SASG
                console.debug("Using RBLA-SASG aggregator")
                return FedAggregator_RBLA_SASG(args)
            case _:
                raise ValueError(f"Unsupported aggregation method: {args.method}")
        return
