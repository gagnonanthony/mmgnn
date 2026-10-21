#!/usr/bin/env python3
"""Visualize node embeddings on cortical surfaces."""

import argparse
import coloredlogs
import logging
from pathlib import Path

from mmgnn.viz.surface import (
    build_group_node_embeddings,
    build_subject_node_embeddings,
    load_subject_ids,
    plot_node_embedding_surfaces,
)


LOGGER = logging.getLogger(__name__)


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize node embeddings as cortical surface figures.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--explanation_dir",
        type=str,
        required=True,
        help=(
            "Directory created by `mmgnn.cli.inference` "
            "(contains subjects/* with node_embeddings.npy)."
        ),
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more subject IDs to visualize, OR a single path to a text file "
            "containing one subject ID per line."
        ),
    )
    parser.add_argument(
        "--atlas",
        type=str,
        required=True,
        help=(
            "Either a built-in yabplot atlas name (e.g. 'aparc', 'schaefer400') OR a "
            "directory with fsLR atlas files (*.dlabel.nii + matching *.tsv). "
            "Custom directories include cortical and subcortical regions (including "
            "cerebellum and brainstem) in connectome node layout."
        ),
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean_signed",
        choices=["mean_abs", "mean_signed"],
        help="Aggregation across subjects when more than one subject is provided.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help=(
            "Output directory, or a filename stem used as "
            "<stem>_dim000.png, <stem>_dim001.png, ..."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, will print detailed progress.",
    )
    return parser


def main() -> None:
    args = argparser().parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    data_dir = Path(args.explanation_dir)
    output_path = Path(args.output_path)

    subjects_arg = [str(value) for value in args.subjects]
    subjects_path = Path(subjects_arg[0])
    if len(subjects_arg) == 1 and subjects_path.exists() and subjects_path.is_file():
        subject_ids = load_subject_ids(subjects_path)
        subjects_label = subjects_path.stem
    else:
        subject_ids = subjects_arg
        subjects_label = "subjects"

    if len(subject_ids) == 1:
        node_embeddings = build_subject_node_embeddings(
            data_dir=data_dir,
            subject_id=subject_ids[0],
        )
        title = f"{subject_ids[0]} (embeddings)"
    else:
        node_embeddings = build_group_node_embeddings(
            data_dir=data_dir,
            subject_ids=subject_ids,
            aggregate=args.aggregate,
        )
        title = f"{subjects_label} (embeddings, {args.aggregate}, n={len(subject_ids)})"

    saved_paths = plot_node_embedding_surfaces(
        node_embeddings=node_embeddings,
        atlas=str(args.atlas),
        output_path=output_path,
        title_prefix=title,
    )
    LOGGER.info("Saved %d embedding surface figure(s) under %s", len(saved_paths), output_path)


if __name__ == "__main__":
    main()
