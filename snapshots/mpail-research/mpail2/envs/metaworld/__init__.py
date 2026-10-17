"""Meta-World environments for MPAIL2.

Heavy dependencies (MuJoCo, ``metaworld``) load only when you import submodules, e.g.:

- ``from mpail2.envs.metaworld.env_factory import setup_metaworld_environment``
- ``from mpail2.envs.metaworld.mpail_obs import MetaWorldMPAILEnv``

Lightweight symbols (numpy only) live in :mod:`mpail2.envs.metaworld.mw_obs`.
"""

# Intentionally avoid importing env_factory / mpail_obs here so that
# ``from mpail2.envs.metaworld.mw_obs import ...`` stays lightweight for learner presets.

__all__: list[str] = []
