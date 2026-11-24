import pettingzoo

_register = getattr(pettingzoo, "register", None)
if _register is not None:
    from mpe2.all_modules import mpe_environments

__version__ = "1.2.0"


def register_mpe2_envs() -> None:
    """Register the MPE environments on PettingZoo versions that provide a registry."""
    if _register is None:
        return

    for env_id, module in mpe_environments.items():
        _register("aec", env_id, module.env)
        _register("parallel", env_id, module.parallel_env)


register_mpe2_envs()

try:
    import sys

    from farama_notifications import notifications

    if "mpe2" in notifications and __version__ in notifications["mpe2"]:
        print(notifications["mpe2"][__version__], file=sys.stderr)
except Exception:  # nosec
    pass
