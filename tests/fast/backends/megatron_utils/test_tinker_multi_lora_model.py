from argparse import Namespace

import pytest

from miles.backends.megatron_utils.tinker_backend.model import _skip_untrained_mtp_adapter


@pytest.mark.parametrize(
    ("prefix", "name"),
    [
        ("language_model.mtp.layers.0.transformer_layer.mlp", "linear_fc1"),
        ("mtp.layers.0.transformer_layer.self_attention", "linear_qkv"),
    ],
)
def test_skips_mtp_adapters_without_mtp_training(prefix, name):
    args = Namespace(enable_mtp_training=False)

    assert _skip_untrained_mtp_adapter(args, name, prefix)


def test_keeps_decoder_adapters_without_mtp_training():
    args = Namespace(enable_mtp_training=False)

    assert not _skip_untrained_mtp_adapter(
        args,
        "linear_fc1",
        "language_model.decoder.layers.0.mlp",
    )


def test_keeps_mtp_adapters_when_mtp_training_is_enabled():
    args = Namespace(enable_mtp_training=True)

    assert not _skip_untrained_mtp_adapter(
        args,
        "linear_fc1",
        "language_model.mtp.layers.0.transformer_layer.mlp",
    )
