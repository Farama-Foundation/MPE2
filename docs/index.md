---
firstpage:
lastpage:
---

```{figure} _static/img/mpe-simple-tag.gif
   :alt: GIF for MPE2 simple_tag environment
   :width: 300
```

```{project-heading}
A set of communication oriented environment
```

**Basic example:**

```{code-block} python

from mpe2 import simple_push_v3

env = simple_push_v3.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
    
env.close()
```

```{toctree}
:hidden:
:caption: Introduction

mpe2
```

```{toctree}
:hidden:
:caption: Environments

environments/simple
environments/simple_adversary
environments/simple_crypto
environments/simple_push
environments/simple_reference
environments/simple_speaker_listener
environments/simple_spread
environments/simple_tag
environments/simple_world_comm
```



```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/MPE2>
release_notes/index
Contribute to the Docs <https://github.com/Farama-Foundation/MPE2/blob/main/docs/README.md>
```
