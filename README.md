# Mucko: Multi-Layer Cross-Modal Knowledge Reasoning for Fact-based Visual Question Answering

This is a Pytorch implementation for [Mucko: Multi-Layer Cross-Modal Knowledge Reasoning for Fact-based Visual Question Answering (IJCAI 2020)](https://ijcai20.org).   
>NOTE: The offical publication has not been published!

# Requirements
1. Install Python 3.7.
2. Install PyTorch 1.2.
3. Install other dependency packages.
4. Clone this repository and enter the root directory of it.  
```
git clone https://github.com/astro-zihao/mucko.git
```

# Usage
For training the model
```
CUDA_VISIBLE_DEVICES=0 python train.py --config-yml exp_fvqa/exp.yml --cpu-workers 8 --gpus 0 --save-dirpath fvqa/exp_data/checkpoints
```

- config-yml: Path to a config file listing reader, model and solver parameters.
- cpu-workers: Number of CPU workers for dataloader.
- save-dirpath: Path of directory to create checkpoint directory and save checkpoints.
- load-pthpath: To continue training, path to .pth file of saved checkpoint.
- validate: Whether to validate on val split after every epoch.



# Bibtex
> @inproceedings{zhu2020mucko,  
    title={Mucko: Multi-Layer Cross-Modal Knowledge Reasoning for Fact-based Visual Question Answering,  
    author={Zhu, Zihao and Yu, Jing and Sun, Yajing and Hu, Yue and Wang, Yujing and Wu, Qi},  
    booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},  
    year={2020}  
    }

# Acknowledgement
Part of this code uses components from [DualVD](https://github.com/JXZe/DualVD). We thank authors for releasing their code.