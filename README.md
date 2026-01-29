# Just on Time

## Installation

```bash
conda create -n dllm python=3.10 -y
conda activate dllm

conda install cuda=12.4 -c nvidia
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

pip install -e .

git submodule update --init --recursive

pip install -e "lm-evaluation-harness[ifeval,math]"
```

### How to run 

```bash
bash scripts/eval_configs.sh
```

You will get best model results for dream

If you want to change parameters, change configurations lines in the eval_configs.sh

### Results

You will get two files:

1. `<model_name>_<benchmark>_efficiency_<config>_<timestamp>.json`: Contains efficiency metrics.
2. `<model_name>_<benchmark>_lm_eval_<config>_<timestamp>`: Directory containing evaluation results.
