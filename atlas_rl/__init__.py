"""Atlas-RL: domain-coherent verifiable RL environments (DevOps/SRE) + GRPO training.

Import side effect: registers all built-in environments in the global registry.
"""

__version__ = "0.1.0"

from atlas_rl.core.registry import REGISTRY, get_env, list_envs  # noqa: F401
import atlas_rl.envs  # noqa: F401  (triggers registration)
