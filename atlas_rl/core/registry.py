"""Global environment registry."""

from __future__ import annotations

from atlas_rl.core.env import AtlasEnv

REGISTRY: dict[str, AtlasEnv] = {}


def register(env_cls: type[AtlasEnv]) -> type[AtlasEnv]:
    env = env_cls()
    if env.env_id in REGISTRY:
        raise ValueError(f"duplicate env_id: {env.env_id}")
    REGISTRY[env.env_id] = env
    return env_cls


def get_env(env_id: str) -> AtlasEnv:
    if env_id not in REGISTRY:
        raise KeyError(f"unknown env '{env_id}'. known: {sorted(REGISTRY)}")
    return REGISTRY[env_id]


def list_envs() -> list[str]:
    return sorted(REGISTRY)
