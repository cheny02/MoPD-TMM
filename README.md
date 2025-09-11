# MoPD: Mixture-of-Prompts Distillation for Vision-Language Models  [IEEE Transactions on Multimedia]

> [**MoPD: Mixture-of-Prompts Distillation for Vision-Language Models**](https://arxiv.org/abs/2412.19087)<br>
> Yang Chen, Shuai Fu, Yu Zhang

# Training and Testing

We provide bash scripts in [scripts/](../scripts) for training and testing MoPD.
Make sure to update the `DATA` variable with dataset path in the script file and run the commands from the main directory `/scripts/`.
Below we provide training and testing instructions for ATLaS.

Make sure conda is installed properly.

The code is built on top of [CoOp](https://github.com/KaiyangZhou/CoOp) which extensively use the toolbox  [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch). So you need to install the dasll environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation). After that, run ```pip install -r requirements.txt``` to install a few more packages.

Follow [Dataset](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to install the datasets.

## How to Run
We provide the running scripts in scripts/run, which allow you to reproduce the results

## Acknowledgement
Our implementation is based on the [CoOp](https://github.com/KaiyangZhou/CoOp) and [CoCoOp](https://github.com/KaiyangZhou/CoOp).

## Citation
```bibtex
If you find our paper of codebase useful, please consider citing us as:
    @article{chen2024mopdmixtureofpromptsdistillationvisionlanguage,
        title={MoPD: Mixture-of-Prompts Distillation for Vision-Language Models}, 
        author={Yang Chen and Shuai Fu and Yu Zhang},
        year={2024},
        eprint={2412.19087},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2412.19087}, 
    }
```
