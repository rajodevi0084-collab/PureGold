"""Reinforcement learning training scaffolding.

This module reuses the predictive data interface to prepare feature
matrices and then executes lightweight hyper-parameter sweeps for
prototype RL agents. The implementation focuses on providing a
repeatable structure for experimenting with different reward functions
and configuration sweeps while logging results to MLflow.
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import mlflow
import numpy as np
import pandas as pd

from models.predictive.train import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CONFIG_PATH,
    configure_mlflow,
    load_config,
    prepare_supervised_dataset,
)

logger = logging.getLogger(__name__)

RewardFunction = Callable[[np.ndarray, np.ndarray, pd.DataFrame], np.ndarray]


def _ensure_mapping(obj: Mapping[str, Any] | None, label: str) -> Mapping[str, Any]:
    if obj is None:
        return {}
    if not isinstance(obj, MutableMapping):
        raise TypeError(f"Expected '{label}' to be a mapping")
    return obj


def build_reward_function(definition: Mapping[str, Any]) -> RewardFunction:
    """Create a reward function callable based on configuration."""

    kind = definition.get("kind")
    if not kind:
        raise ValueError("Reward definition must include 'kind'")

    if kind == "risk_adjusted":
        transaction_cost = float(definition.get("transaction_cost", 0.0))
        risk_penalty = float(definition.get("risk_penalty", 0.0))

        def reward_fn(actions: np.ndarray, returns: np.ndarray, _: pd.DataFrame) -> np.ndarray:
            turnover = np.abs(np.diff(actions, prepend=0.0))
            pnl = actions * returns - transaction_cost * turnover
            risk_penalty_term = risk_penalty * np.square(actions)
            return pnl - risk_penalty_term

    elif kind == "directional":
        positive = float(definition.get("positive_reward", 1.0))
        negative = float(definition.get("negative_reward", -1.0))
        threshold = float(definition.get("threshold", 0.0))

        def reward_fn(actions: np.ndarray, returns: np.ndarray, _: pd.DataFrame) -> np.ndarray:
            comparison = np.sign(actions) == np.sign(returns - threshold)
            return np.where(comparison, positive, negative)

    else:
        raise ValueError(f"Unsupported reward kind: {kind}")

    return reward_fn


def sample_actions(
    window: pd.DataFrame,
    feature_columns: Sequence[str],
    rng: np.random.Generator,
    hyperparams: Mapping[str, Any],
) -> np.ndarray:
    """Generate placeholder trading actions informed by feature momentum."""

    if feature_columns:
        features = window[feature_columns].to_numpy(dtype=float, copy=False)
        weights = np.linspace(1.0, 0.2, num=features.shape[1], endpoint=True)
        base_signal = features @ weights
    else:
        base_signal = np.zeros(len(window))

    noise_scale = float(hyperparams.get("entropy_coef", 0.01)) + 1e-3
    sensitivity = float(hyperparams.get("learning_rate", 0.001))
    raw_signal = sensitivity * base_signal + rng.normal(scale=noise_scale, size=len(base_signal))
    actions = np.clip(np.round(np.tanh(raw_signal)), -1.0, 1.0)
    return actions


def simulate_episode(
    rng: np.random.Generator,
    market_df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    reward_fn: RewardFunction,
    episode_length: int,
    gamma: float,
    hyperparams: Mapping[str, Any],
) -> Dict[str, float]:
    """Simulate an episode using a stochastic policy to establish baselines."""

    if len(market_df) < episode_length:
        raise ValueError("Episode length exceeds available market history")

    start_idx = int(rng.integers(0, len(market_df) - episode_length + 1))
    window = market_df.iloc[start_idx : start_idx + episode_length]
    actions = sample_actions(window, feature_columns, rng, hyperparams)
    returns = window[target_column].to_numpy(dtype=float, copy=False)
    rewards = reward_fn(actions, returns, window)

    discount_factors = gamma ** np.arange(len(rewards))
    discounted_reward = float(np.dot(rewards, discount_factors))
    cumulative_reward = float(np.sum(rewards))
    mean_reward = float(np.mean(rewards))
    turnover = float(np.sum(np.abs(np.diff(actions, prepend=0.0))))
    strategy_returns = actions * returns
    sharpe = float(
        np.mean(strategy_returns) / np.std(strategy_returns, ddof=1)
    ) if len(strategy_returns) > 1 and np.std(strategy_returns, ddof=1) > 1e-12 else 0.0
    directional_accuracy = float(np.mean(np.sign(actions) == np.sign(returns)))
    hit_rate = float(np.mean(rewards > 0.0))

    return {
        "cumulative_reward": cumulative_reward,
        "discounted_reward": discounted_reward,
        "mean_reward": mean_reward,
        "turnover": turnover,
        "sharpe": sharpe,
        "directional_accuracy": directional_accuracy,
        "hit_rate": hit_rate,
    }


def aggregate_episode_metrics(episodes: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Aggregate per-episode metrics into dataset-level KPIs."""

    if not episodes:
        return {}
    aggregated: Dict[str, float] = {}
    for key in episodes[0].keys():
        values = np.array([episode[key] for episode in episodes], dtype=float)
        aggregated[f"mean_{key}"] = float(np.mean(values))
        aggregated[f"std_{key}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return aggregated


def iterate_hyperparameters(hyperparameters: Mapping[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    """Yield every combination of hyper-parameter values."""

    if not hyperparameters:
        yield {}
        return
    keys = list(hyperparameters.keys())
    values = [list(hyperparameters[key]) for key in keys]
    for combination in product(*values):
        yield {key: value for key, value in zip(keys, combination)}


def run_hyperparameter_sweep(
    pandas_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    feature_set_name: str,
    reward_definitions: Sequence[Mapping[str, Any]],
    hyperparameters: Mapping[str, Iterable[Any]],
    artifact_dir: Path,
    episodes: int,
    episode_length: int,
    seed: int,
) -> None:
    """Execute reward-aware hyper-parameter sweeps with metric logging."""

    market_df = pandas_frame[feature_columns + [target_column]].copy()
    rng = np.random.default_rng(seed)

    for reward_def in reward_definitions:
        reward_name = reward_def.get("name", reward_def.get("kind", "reward"))
        reward_fn = build_reward_function(reward_def)

        for idx, hyperparam_set in enumerate(iterate_hyperparameters(hyperparameters)):
            gamma = float(hyperparam_set.get("gamma", 0.99))
            mlflow_params = {
                **{f"hp_{key}": value for key, value in hyperparam_set.items()},
                "reward": reward_name,
                "feature_set": feature_set_name,
                "episode_length": episode_length,
                "episodes": episodes,
            }

            run_name = f"RL_{reward_name}_{idx:03d}"
            with mlflow.start_run(run_name=run_name):
                logger.info("Running RL sweep %s with params %s", run_name, hyperparam_set)
                episode_metrics: List[Mapping[str, float]] = []
                for _ in range(episodes):
                    episode_metrics.append(
                        simulate_episode(
                            rng=rng,
                            market_df=market_df,
                            feature_columns=feature_columns,
                            target_column=target_column,
                            reward_fn=reward_fn,
                            episode_length=episode_length,
                            gamma=gamma,
                            hyperparams=hyperparam_set,
                        )
                    )

                aggregated = aggregate_episode_metrics(episode_metrics)
                aggregated["episodes"] = episodes
                aggregated["episode_length"] = episode_length

                mlflow.log_params(mlflow_params)
                for key, value in aggregated.items():
                    mlflow.log_metric(key, float(value))

                run_artifact_dir = artifact_dir / "rl" / reward_name
                run_artifact_dir.mkdir(parents=True, exist_ok=True)
                results_path = run_artifact_dir / f"{run_name}.json"
                with results_path.open("w", encoding="utf-8") as handle:
                    json.dump({"hyperparameters": hyperparam_set, "metrics": aggregated}, handle, indent=2)
                mlflow.log_artifact(results_path, artifact_path="results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL training scaffolding")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Predictive YAML config")
    parser.add_argument("--data-path", type=Path, required=False, help="Dataset override path")
    parser.add_argument("--feature-set", type=str, default=None, help="Feature set to engineer")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR, help="Artifact directory")
    parser.add_argument("--experiment", type=str, default=None, help="MLflow experiment name")
    parser.add_argument("--episodes", type=int, default=None, help="Number of evaluation episodes")
    parser.add_argument("--episode-length", type=int, default=None, help="Episode length in timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    config = load_config(args.config)
    data_cfg = _ensure_mapping(config.get("data"), "data")
    rl_cfg = _ensure_mapping(config.get("rl"), "rl")

    data_path = args.data_path or Path(data_cfg.get("path", ""))
    if not data_path:
        raise ValueError("Data path must be supplied either via --data-path or config.data.path")

    feature_set = args.feature_set or rl_cfg.get("default_feature_set")
    if not feature_set:
        raise ValueError("Feature set must be specified via --feature-set or rl.default_feature_set")

    dataset = prepare_supervised_dataset(config, data_path, feature_set)
    pandas_frame = dataset.frame.to_pandas()

    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = args.experiment or config.get("mlflow", {}).get("experiment", "predictive_baselines")
    configure_mlflow(artifact_dir, experiment_name)

    reward_definitions = rl_cfg.get("reward_definitions", [])
    if not reward_definitions:
        raise ValueError("Configuration must define rl.reward_definitions")

    hyperparameters = rl_cfg.get("hyperparameters", {})
    episodes = args.episodes or int(rl_cfg.get("evaluation_episodes", 10))
    episode_length = args.episode_length or int(rl_cfg.get("episode_length", 64))

    run_hyperparameter_sweep(
        pandas_frame=pandas_frame,
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        feature_set_name=feature_set,
        reward_definitions=reward_definitions,
        hyperparameters=hyperparameters,
        artifact_dir=artifact_dir,
        episodes=episodes,
        episode_length=episode_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
