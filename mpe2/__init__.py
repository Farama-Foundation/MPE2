__version__ = "1.2.0"


def _register_environments() -> None:
    """Register the MPE environments on PettingZoo versions that provide a registry."""
    import pettingzoo

    register = getattr(pettingzoo, "register", None)
    if register is None:
        return

    from mpe2.all_modules import mpe_environments

    for env_id, module in mpe_environments.items():
        register("aec", env_id, module.env)
        register("parallel", env_id, module.parallel_env)


_register_environments()

try:
    import sys

    from farama_notifications import notifications

    if "mpe2" in notifications and __version__ in notifications["mpe2"]:
        print(notifications["mpe2"][__version__], file=sys.stderr)
except Exception:  # nosec
    pass
