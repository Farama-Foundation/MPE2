---
firstpage:
lastpage:
---

```{project-logo} _static/img/mpe2-text.png
:alt: MPE2 Logo
```

```{project-heading}
A set of communication-oriented environments
```

```{figure} _static/img/mpe-simple-tag.gif
   :alt: GIF for MPE2 simple_tag environment
   :width: 300
```

Multi Particle Environments 2 (MPE2) is collection of environments where particle agents can (sometimes) move, communicate, see each other, push each other around, and interact with fixed landmarks.

These environments are originally from [OpenAI’s MPE codebase](https://github.com/openai/multiagent-particle-envs), with several minor fixes, mostly related to making the action space discrete by default, making the rewards consistent and cleaning up the observation space of certain environments. MPE2 additionally includes 3 new environments.

## Installation

The unique dependencies for this set of environments can be installed via:

```bash
pip install mpe2
```


## Example

To launch a [Simple Push](/environments/simple_push/) environment with random agents:

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


## Citation

The MPE environments were originally described in the following work:

```
@article{mordatch2017emergence,
  title={Emergence of Grounded Compositional Language in Multi-Agent Populations},
  author={Mordatch, Igor and Abbeel, Pieter},
  journal={arXiv preprint arXiv:1703.04908},
  year={2017}
}
```

But were first released as a part of this work:

```
@article{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
```

Please cite one or both of these if you use these environments in your research.


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
environments/simple_formation
environments/simple_line
environments/simple_push
environments/simple_reference
environments/simple_speaker_listener
environments/simple_spread
environments/simple_tag
environments/simple_world_comm
environments/collect_treasure
```


```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/MPE2>
Contribute to the Docs <https://github.com/Farama-Foundation/MPE2/blob/main/docs/README.md>
```
