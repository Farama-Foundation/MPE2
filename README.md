# MPE2

## Installation Steps

```bash
# create venv
conda create -n mpe2_env python=3.10
conda activate mpe2_env

# clone repo
git clone https://github.com/Farama-Foundation/MPE2.git

# for development
python3 -m pip install -e .[testing, documentation]

# for usage
python3 -m pip install -e .
```