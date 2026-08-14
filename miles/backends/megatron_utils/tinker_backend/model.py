"""Model-side helpers for the multi-LoRA slot table: building the MultiLoRA
megatron object and trimming max-rank-padded LoRA exports to an adapter's
real rank (weight sync and HF PEFT export both require it)."""

from argparse import Namespace

import torch


def _skip_untrained_mtp_adapter(args: Namespace, name: str | None, prefix: str | None) -> bool:
    """Keep auxiliary MTP layers frozen unless MTP training was explicitly enabled.

    Some checkpoints, including Qwen3.5, advertise an MTP block even during
    ordinary language-model training.  Attaching LoRA to that block is both
    unnecessary (it receives no MTP loss) and cannot be exported to rollout
    engines by Megatron Bridge's PEFT converter.
    """
    full_name = f"{prefix}.{name}" if prefix else name or ""
    return not getattr(args, "enable_mtp_training", False) and "mtp" in full_name.split(".")


def create_multi_lora_instance(args: Namespace):
    """Create a MultiLoRA instance from training args."""
    from megatron.bridge.peft.multi_lora import MultiLoRA

    from miles.backends.megatron_utils.lora_utils import convert_target_modules_to_megatron

    lora_type_name = getattr(args, "lora_type", "lora").lower()
    if lora_type_name == "canonical_lora":
        from megatron.bridge.peft.canonical_lora import CanonicalLoRA

        lora_cls = CanonicalLoRA
    else:
        from megatron.bridge.peft.lora import LoRA

        lora_cls = LoRA

    class TrainingAwareMultiLoRA(MultiLoRA):
        def transform(self, module, name=None, prefix=None):
            if _skip_untrained_mtp_adapter(args, name, prefix):
                return module
            return super().transform(module, name=name, prefix=prefix)

    # exclude_modules was already folded into target_modules during arg validation.
    return TrainingAwareMultiLoRA(
        target_modules=convert_target_modules_to_megatron(args.target_modules, lora_type=lora_cls),
        n_adapters=args.multi_lora_n_adapters,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=getattr(args, "lora_dropout", 0.0),
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )


def slice_lora_to_rank(hf_name: str, tensor: torch.Tensor, adapter_rank: int) -> torch.Tensor:
    """Trim a max-rank-padded LoRA tensor to ``adapter_rank`` on the rank axis, addressed
    from the end so packed grouped-expert exports are not sliced on the expert axis."""
    if "lora_A" in hf_name:
        rank_dim = tensor.ndim - 2
        if adapter_rank < tensor.shape[rank_dim]:
            remainder = tensor.narrow(rank_dim, adapter_rank, tensor.shape[rank_dim] - adapter_rank)
            assert remainder.abs().max() == 0, (
                f"lora_A padded dims are non-zero: {hf_name}, "
                f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
            )
            return tensor.narrow(rank_dim, 0, adapter_rank)
        return tensor
    if "lora_B" in hf_name:
        rank_dim = tensor.ndim - 1
        if adapter_rank < tensor.shape[rank_dim]:
            remainder = tensor.narrow(rank_dim, adapter_rank, tensor.shape[rank_dim] - adapter_rank)
            assert remainder.abs().max() == 0, (
                f"lora_B padded dims are non-zero: {hf_name}, "
                f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
            )
            return tensor.narrow(rank_dim, 0, adapter_rank)
        return tensor
    return tensor
