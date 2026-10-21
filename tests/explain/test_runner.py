"""Unit tests for mmgnn.explain.runner.run_ablation."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from mmgnn.explain.runner import run_ablation
from mmgnn.models.utils import build_encoder, build_model
from tests.explain.helpers import LATENT_DIM, NUM_NODES, make_graph, make_z

SUBJECT_IDS = ("sub-001", "sub-002")


def write_embeddings(embeddings_dir: Path, subject_id: str, z: torch.Tensor) -> None:
    subject_dir = embeddings_dir / "subjects" / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    np.save(subject_dir / "node_embeddings.npy", z.detach().cpu().numpy())


def model_config() -> dict[str, Any]:
    return {
        "hidden_dim": LATENT_DIM,
        "num_heads": 4,
        "num_layers": 1,
        "conv_type": "gine",
        "act_func": "relu",
        "norm_type": "batch",
        "dropout": 0.0,
        "edge_dropout": 0.0,
        "residual": False,
        "model_type": "gae",
        "edge_presence_decoder_type": "inner_product",
        "edge_attr_decoder_type": "mlp",
        "decode_node_features": True,
        "recon_edge_presence_weight": 1.0,
        "recon_node_attr_weight": 1.0,
        "recon_edge_attr_weight": 1.0,
        "recon_node_loss_type": "mse",
        "recon_edge_attr_loss_type": "mse",
        "neg_sampling_ratio": 1.0,
    }


def ablation_config(**overrides: Any) -> dict[str, Any]:
    config = {"seed": 0, "greedy_topk": 2, "run_greedy": True}
    config.update(overrides)
    return config


def _write_checkpoint(path: Path, graph: Data) -> None:
    """Save a GAE whose input sizes match ``graph`` and whose latent size matches embeddings."""
    config = model_config()
    num_node_features = int(graph.x.size(-1))
    num_edge_features = int(graph.edge_attr.size(-1))
    encoder = build_encoder(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        model_config=config,
    )
    model = build_model(
        encoder=encoder,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        model_config=config,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "model_type": config["model_type"],
                "edge_presence_decoder_type": config["edge_presence_decoder_type"],
            },
        },
        path,
    )


class TestRunAblation:
    def test_empty_graphs_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No graphs"):
            run_ablation(
                graphs=[],
                checkpoint_path=tmp_path / "ckpt.pt",
                validated_model_config=model_config(),
                embeddings_dir=tmp_path / "emb",
                ablation_config=ablation_config(),
                output_dir=tmp_path / "out",
                device=torch.device("cpu"),
            )

    def test_happy_path_writes_artifacts(self, tmp_path: Path):
        embeddings_dir = tmp_path / "emb"
        output_dir = tmp_path / "out"
        z0 = make_z()
        graphs = [make_graph(subject_id=sid) for sid in SUBJECT_IDS]
        for sid in SUBJECT_IDS:
            write_embeddings(embeddings_dir, sid, z0)
        checkpoint_path = tmp_path / "ckpt.pt"
        _write_checkpoint(checkpoint_path, graphs[0])

        summaries, decoder_type, checkpoint_meta = run_ablation(
            graphs=graphs,
            checkpoint_path=checkpoint_path,
            validated_model_config=model_config(),
            embeddings_dir=embeddings_dir,
            ablation_config=ablation_config(),
            output_dir=output_dir,
            device=torch.device("cpu"),
        )

        assert decoder_type == "inner_product"
        assert checkpoint_meta == {"model_type": "gae"}
        assert len(summaries) == 2
        assert {s["subject_id"] for s in summaries} == set(SUBJECT_IDS)
        for sid in SUBJECT_IDS:
            assert (output_dir / "subjects" / sid / "summary.json").is_file()
        assert (output_dir / "dimension_importance.csv").is_file()
        assert (output_dir / "greedy_ablation.csv").is_file()
        assert (output_dir / "joint_topk_ablation.json").is_file()

    def test_run_greedy_false_skips_greedy_csv(self, tmp_path: Path):
        embeddings_dir = tmp_path / "emb"
        output_dir = tmp_path / "out"
        z0 = make_z()
        graphs = [make_graph(subject_id=SUBJECT_IDS[0])]
        write_embeddings(embeddings_dir, SUBJECT_IDS[0], z0)
        checkpoint_path = tmp_path / "ckpt.pt"
        _write_checkpoint(checkpoint_path, graphs[0])

        run_ablation(
            graphs=graphs,
            checkpoint_path=checkpoint_path,
            validated_model_config=model_config(),
            embeddings_dir=embeddings_dir,
            ablation_config=ablation_config(run_greedy=False, greedy_topk=1),
            output_dir=output_dir,
            device=torch.device("cpu"),
        )

        assert (output_dir / "subjects" / SUBJECT_IDS[0] / "summary.json").is_file()
        assert not (output_dir / "greedy_ablation.csv").exists()
        assert (output_dir / "dimension_importance.csv").is_file()
        assert (output_dir / "joint_topk_ablation.json").is_file()

    def test_embedding_dim_mismatch_raises(self, tmp_path: Path):
        embeddings_dir = tmp_path / "emb"
        bad_z = make_z(latent_dim=LATENT_DIM + 1)
        graphs = [make_graph(subject_id=SUBJECT_IDS[0])]
        write_embeddings(embeddings_dir, SUBJECT_IDS[0], bad_z)
        checkpoint_path = tmp_path / "ckpt.pt"
        _write_checkpoint(checkpoint_path, graphs[0])

        with pytest.raises(ValueError, match="embedding dim"):
            run_ablation(
                graphs=graphs,
                checkpoint_path=checkpoint_path,
                validated_model_config=model_config(),
                embeddings_dir=embeddings_dir,
                ablation_config=ablation_config(),
                output_dir=tmp_path / "out",
                device=torch.device("cpu"),
            )

    def test_node_count_mismatch_raises(self, tmp_path: Path):
        embeddings_dir = tmp_path / "emb"
        bad_z = make_z(num_nodes=NUM_NODES - 1)
        graphs = [make_graph(num_nodes=NUM_NODES, subject_id=SUBJECT_IDS[0])]
        write_embeddings(embeddings_dir, SUBJECT_IDS[0], bad_z)
        checkpoint_path = tmp_path / "ckpt.pt"
        _write_checkpoint(checkpoint_path, graphs[0])

        with pytest.raises(ValueError, match="embeddings have"):
            run_ablation(
                graphs=graphs,
                checkpoint_path=checkpoint_path,
                validated_model_config=model_config(),
                embeddings_dir=embeddings_dir,
                ablation_config=ablation_config(),
                output_dir=tmp_path / "out",
                device=torch.device("cpu"),
            )
