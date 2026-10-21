"""
Utility functions to build model components based on configurations
"""

import logging
from typing import Mapping, Any

from mmgnn.models import GAE, Encoder, VGAE

LOGGER = logging.getLogger(__name__)


def build_encoder(
    *,
    num_node_features: int,
    num_edge_features: int,
    model_config: Mapping[str, Any],
) -> Encoder:
    encoder_kwargs = {
        key: model_config[key]
        for key in (
            "hidden_dim",
            "num_heads",
            "num_layers",
            "conv_type",
            "act_func",
            "norm_type",
            "dropout",
            "edge_dropout",
            "residual",
        )
    }

    if encoder_kwargs["conv_type"] != "gat":
        encoder_kwargs["num_heads"] = None

    return Encoder(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        **encoder_kwargs,
    )


def build_model(
    *,
    encoder: Encoder,
    num_node_features: int,
    num_edge_features: int,
    model_config: Mapping[str, Any],
) -> GAE:
    edge_attr_out_dim = model_config.get("edge_attr_out_dim")
    if edge_attr_out_dim is None and num_edge_features > 0:
        edge_attr_out_dim = num_edge_features

    node_out_dim = model_config.get("node_out_dim")
    if node_out_dim is None:
        node_out_dim = num_node_features

    if model_config["edge_presence_decoder_type"] == "mlp" and model_config.get("decoder_hidden_dim") is None:
        LOGGER.debug("Using encoder hidden_dim as decoder_hidden_dim for edge presence MLP")

    model_type = str(model_config.get("model_type", "gae")).lower()
    common_kwargs = {
        "encoder": encoder,
        "edge_presence_decoder_type": model_config["edge_presence_decoder_type"],
        "edge_attr_out_dim": edge_attr_out_dim,
        "edge_attr_decoder_type": model_config["edge_attr_decoder_type"],
        "decode_node_features": bool(model_config.get("decode_node_features", True)),
        "node_out_dim": node_out_dim,
        "decoder_hidden_dim": model_config.get("decoder_hidden_dim"),
        "dropout": float(model_config.get("dropout", 0.0)),
    }

    if model_type == "vgae":
        return VGAE(
            kl_weight=float(model_config.get("kl_weight", 1.0)),
            **common_kwargs,
        )
    if model_type == "gae":
        return GAE(**common_kwargs)
    raise ValueError(f"Unsupported model_type '{model_type}'. Supported values: 'gae', 'vgae'")
