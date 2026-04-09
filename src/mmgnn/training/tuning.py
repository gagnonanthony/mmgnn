"""
Hyperparameter tuning utilities for MMGNN training.
Leverages Ray Tune for efficient hyperparameter search and management combined
with OptunaSearch
"""

import copy
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

try:
    import ray
    from ray import tune
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.optuna import OptunaSearch
    _RAY_TUNE_IMPORT_ERROR = None
except Exception as exc:
    ray = None
    tune = None
    ConcurrencyLimiter = None
    OptunaSearch = None
    _RAY_TUNE_IMPORT_ERROR = exc


def _ensure_ray_tune_available() -> None:
    """Raise a clear error when ray[tune] optional dependencies are missing."""
    if ray is None or tune is None or ConcurrencyLimiter is None or OptunaSearch is None:
        raise ImportError(
            "Ray Tune is not available. Install optional dependencies with: pip install \"ray[tune]\""
        ) from _RAY_TUNE_IMPORT_ERROR


def _safe_trial_id() -> str:
    """Get the current Ray trial id when available, otherwise return a local fallback."""
    if tune is None:
        return "local_trial"

    try:
        context = tune.get_context()
    except Exception:
        context = None

    if context is None:
        return "local_trial"

    trial_id = context.get_trial_id()
    if trial_id is None:
        return "local_trial"
    return str(trial_id)


def _extract_tuning_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Extract scalar metrics from run_experiment outputs for Ray reporting."""
    metrics: Dict[str, float] = {}

    if "best_val_loss" in results and results["best_val_loss"] is not None:
        metrics["val_loss"] = float(results["best_val_loss"])

    if "best_epoch" in results and results["best_epoch"] is not None:
        metrics["best_epoch"] = float(results["best_epoch"])

    val_results = results.get("val_results") if isinstance(results, dict) else None
    if isinstance(val_results, dict):
        for key in ("accuracy", "f1", "mae", "mse", "pearson"):
            if key in val_results and val_results[key] is not None:
                metrics[f"val_{key}"] = float(val_results[key])

    test_results = results.get("test_results") if isinstance(results, dict) else None
    if isinstance(test_results, dict):
        for key in ("accuracy", "f1", "mae", "mse", "pearson"):
            if key in test_results and test_results[key] is not None:
                metrics[f"test_{key}"] = float(test_results[key])

    return metrics


def _normalize_storage_path(storage_path: str) -> str:
    """Normalize Ray Tune storage path to a URI/path accepted by PyArrow.

    Ray 2.5x resolves storage_path with pyarrow.fs.FileSystem.from_uri,
    which rejects relative local paths such as './ray_results'.
    """
    parsed = urlparse(storage_path)
    if parsed.scheme:
        return storage_path

    # Treat schemeless inputs as local filesystem paths.
    absolute_path = Path(storage_path).expanduser().resolve()
    return absolute_path.as_uri()


def _extract_trial_results_df(
    analysis: Any,
    metric: str,
    save_csv_path: Optional[Path] = None,
) -> Any:
    """Extract trial results into a pandas DataFrame with flattened configs and metrics.

    Args:
        analysis: Ray ExperimentAnalysis object from tune.run.
        metric: Primary metric name used for optimization.
        save_csv_path: Optional path to save the results DataFrame as CSV.

    Returns:
        pandas.DataFrame: A DataFrame with one row per trial, including trial id, config
        parameters, and all reported metrics.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to use results summary. Install with: pip install pandas")

    # Use Ray's built-in results_df which includes all trials and their metrics.
    results_df = analysis.results_df

    # Ensure the primary metric is present and visible.
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in trial results. Available: {results_df.columns.tolist()}")

    # Sort by the optimization metric for easy inspection.
    results_df = results_df.sort_values(by=metric, ascending=True)

    # Save to CSV if requested.
    if save_csv_path:
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_csv_path, index=False)

    return results_df


def make_run_experiment_trainable(
    run_experiment_func: Callable[..., Dict[str, Any]],
    fixed_kwargs: Optional[Dict[str, Any]] = None,
    trial_checkpoint_root: str = "./ray_results/trial_checkpoints",
    default_verbose: bool = False,
    default_generate_artifacts: bool = False,
) -> Callable[[Dict[str, Any]], None]:
    """Build a Ray trainable that runs run_experiment with sampled config values."""
    _ensure_ray_tune_available()

    base_fixed_kwargs = copy.deepcopy(fixed_kwargs or {})

    func_signature = signature(run_experiment_func)
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in func_signature.parameters.values())
    supported_params = set(func_signature.parameters.keys())

    def trainable(sampled_config: Dict[str, Any], fixed_kwargs: Optional[Dict[str, Any]] = None) -> None:
        trial_id = _safe_trial_id()
        trial_checkpoint_dir = Path(trial_checkpoint_root) / f"trial_{trial_id}"
        trial_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        run_kwargs: Dict[str, Any] = copy.deepcopy(base_fixed_kwargs)
        if fixed_kwargs:
            run_kwargs.update(copy.deepcopy(fixed_kwargs))
        run_kwargs.update(sampled_config)

        run_kwargs.setdefault("checkpoint_dir", trial_checkpoint_dir)
        run_kwargs.setdefault("verbose", default_verbose)
        run_kwargs.setdefault("generate_artifacts", default_generate_artifacts)

        if not accepts_kwargs:
            run_kwargs = {k: v for k, v in run_kwargs.items() if k in supported_params}

        results = run_experiment_func(**run_kwargs)
        metrics = _extract_tuning_metrics(results)

        if "val_loss" not in metrics:
            raise ValueError(
                "Ray tuning metric 'val_loss' was not found in run_experiment outputs. "
                "Ensure run_experiment returns best_val_loss or val_results with loss information."
            )

        metrics["trial_checkpoint_dir"] = str(trial_checkpoint_dir)
        tune.report(metrics)

    return trainable


def tune_hyperparameters(
    train_func: Callable[[Dict[str, Any]], Any],
    config: Dict[str, Any],
    num_samples: int = 20,
    max_concurrent_trials: int = 4,
    local_dir: str = "./ray_results",
    storage_path: Optional[str] = None,
    metric: str = "val_loss",
    mode: str = "min",
    resources_per_trial: Optional[Dict[str, float]] = None,
    ray_init_kwargs: Optional[Dict[str, Any]] = None,
    trainable_kwargs: Optional[Dict[str, Any]] = None,
    shutdown_ray: bool = False,
    **tune_run_kwargs: Any,
) -> Dict[str, Any]:
    """
    Runs hyperparameter tuning using Ray Tune with OptunaSearch.

    Args:
        train_func (callable): The training function to optimize. Should accept a config dict as input.
        config (dict): A dictionary specifying the hyperparameter search space.
        num_samples (int): The number of hyperparameter configurations to sample.
        max_concurrent_trials (int): Maximum number of concurrent trials to run.
        local_dir (str): Deprecated alias for the Ray Tune storage directory.
        storage_path (str | None): Directory where Ray Tune will save results.
            If None, local_dir is used for backward compatibility.
        metric (str): Metric name to optimize.
        mode (str): Either "min" or "max" for metric optimization direction.
        resources_per_trial (dict | None): Ray resources per trial.
        ray_init_kwargs (dict | None): Optional kwargs passed to ray.init.
        trainable_kwargs (dict | None): Optional kwargs bound to train_func via tune.with_parameters.
            Use this to pass large Python objects (for example datasets) without capturing them in the
            trainable closure and hitting Ray gRPC registration size limits.
        shutdown_ray (bool): Whether to call ray.shutdown() after finishing.
        tune_run_kwargs: Additional keyword arguments forwarded to ray.tune.run.

    Returns:
        dict: Summary containing best config, metrics, and trial metadata.
    """
    _ensure_ray_tune_available()

    if mode not in {"min", "max"}:
        raise ValueError("mode must be either 'min' or 'max'")

    if resources_per_trial is None:
        resources_per_trial = {"cpu": 2, "gpu": 1}

    if not ray.is_initialized():
        init_kwargs = {"ignore_reinit_error": True}
        init_kwargs.update(ray_init_kwargs or {})
        ray.init(**init_kwargs)

    raw_storage_path = storage_path or local_dir
    normalized_storage_path = _normalize_storage_path(raw_storage_path)

    # Ensure local directories exist when a local path/URI is used.
    parsed_storage = urlparse(normalized_storage_path)
    if parsed_storage.scheme in {"", "file"}:
        local_storage_dir = (
            Path(parsed_storage.path).resolve()
            if parsed_storage.scheme == "file"
            else Path(normalized_storage_path).expanduser().resolve()
        )
        local_storage_dir.mkdir(parents=True, exist_ok=True)

    optuna_search = OptunaSearch(metric=metric, mode=mode)
    search_alg = ConcurrencyLimiter(optuna_search, max_concurrent=max_concurrent_trials)

    run_kwargs = {
        "config": config,
        "num_samples": num_samples,
        "search_alg": search_alg,
        "storage_path": normalized_storage_path,
        "resources_per_trial": resources_per_trial,
        "metric": metric,
        "mode": mode,
    }
    run_kwargs.update(tune_run_kwargs)

    run_train_func = train_func
    if trainable_kwargs:
        run_train_func = tune.with_parameters(train_func, **copy.deepcopy(trainable_kwargs))

    try:
        analysis = tune.run(run_train_func, **run_kwargs)
    finally:
        if shutdown_ray and ray.is_initialized():
            ray.shutdown()

    best_trial = analysis.get_best_trial(metric=metric, mode=mode, scope="all")
    if best_trial is None:
        raise RuntimeError(f"No completed Ray trials found for metric '{metric}'")

    best_config = analysis.get_best_config(metric=metric, mode=mode)
    best_result = analysis.best_result if analysis.best_result is not None else {}
    best_logdir = best_trial.local_path

    # Extract and optionally save trial results summary.
    try:
        parsed_storage = urlparse(normalized_storage_path)
        if parsed_storage.scheme in {"", "file"}:
            local_storage_dir = (
                Path(parsed_storage.path).resolve()
                if parsed_storage.scheme == "file"
                else Path(normalized_storage_path).expanduser().resolve()
            )
        else:
            local_storage_dir = None

        csv_path = (local_storage_dir / "trials_summary.csv") if local_storage_dir else None
        trials_df = _extract_trial_results_df(analysis, metric=metric, save_csv_path=csv_path)
        trials_summary = trials_df.to_dict(orient="records") if csv_path is None else None
    except Exception as exc:
        trials_df = None
        trials_summary = None

    return {
        "best_config": best_config,
        "best_result": best_result,
        "best_metric": best_trial.last_result.get(metric),
        "best_logdir": best_logdir,
        "metric": metric,
        "mode": mode,
        "num_samples": num_samples,
        "max_concurrent_trials": max_concurrent_trials,
        "ray_results_dir": normalized_storage_path,
        "trials_df": trials_df,
        "trials_csv": str(csv_path) if csv_path else None,
    }


