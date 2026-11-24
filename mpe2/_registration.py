"""Register the MPE environments on PettingZoo versions that provide a registry."""

import pettingzoo

register = getattr(pettingzoo, "register", None)
if register is not None:
    from mpe2.all_modules import mpe_environments

    for env_id, module in mpe_environments.items():
        register("aec", env_id, module.env)
        register("parallel", env_id, module.parallel_env)
