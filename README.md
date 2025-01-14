# Can Foundation Models Help Us Achieve Perfect Secrecy? A simple baseline for personal & private ML.

This repository explores the baseline of using in-context learning for personal & private machine learning! We include scripts to download data and evaluate foundation models of various sizes and types across popular personal machine learning benchmarks from the privacy literature. Additional information can be found in the paper: https://arxiv.org/abs/2205.13722. Contributions of additional benchmarks and baselines are welcome in the form of PRs to the benchmarks folder and additions of a data loader. Exciting future questions: 

- Do we see further personalization with more in-context examples and longer contexts? 
- Can we understand the limits of this baseline in a more principled manner? Are the tasks we've evaluated on so far too similar to the pretraining distribution? [Maybe we should construct new privacy benchmarks!](https://arxiv.org/abs/2212.06470)
- Can we enable better in-context learning quality in smaller & open-source models? (Checkout recent work: [Ask Me Anything](https://arxiv.org/abs/2210.02441))
- Other ways of using the FMs -- is it better to generate synthetic data with the FMs and then train locally?

<p align="center"><img width="85%" src="imgs/main_figure.png" /></p>

## Setup

Use the following commands to clone and install this package. We highly recommend you use conda environments.

```
# environment
conda create -n py37 python=3.7
conda activate py37

# installations
git clone git@github.com:hooksz/Focus-Secrecy-Baseline.git
cd Focus-Secrecy-Baseline
pip install -e .
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

cd benchmarks/leaf
git submodule init
git submodule update
```

If you want to run inference with the API models, first obtain your OpenAI inference API key here [OpenAI API](https://openai.com/api/). Then set the environment variables:
```
export OPENAI_API_KEY="<YOUR API KEY>"
```

## Obtain the datasets

Download benchmark datasets to the ``Focus-Secrecy-Baseline/benchmarks/'' directory
```
cd benchmarks/
bash download_data.sh
```

The LEAF Federated Learning benchmark suite provides: Sent140, Reddit, FEMNIST, and CELEB-A. The FedNLP suite provides 20News and MRQA. The FedML suite provides CIFAR-10.
- Sent140, FEMNIST, CelebA, CIFAR-10, and 20News benchmarks are downloaded via the provided download script. 
- [Reddit] Go to benchmarks/leaf/data/reddit/ and follow the download instructions.
- [MRQA] Go to https://github.com/FedML-AI/FedNLP/tree/27f3f97c72e7f206f8937fe6bcbba39ce79fbcd6/data/raw_data_loader/MRQA and run ``python download.py`` using their provided script.

## Run the code

The ``Focus-Secrecy-Baseline/scripts/`` directory provides scripts to run experiments for each benchmark.

For example:
```
bash scripts/cifar.sh
bash scripts/sent140.sh
```

We include examples for running inference with the API in the script files. Note that this requires providing the ```openai_key``` command line argument.

## Citation
Please use the following Bibtex for this work:
```
@misc{arora2022focus,
      title={Can Foundation Models Help Us Achieve Perfect Secrecy?}, 
      author={Simran Arora and Christopher Ré},
      year={2022},
      url={https://arxiv.org/abs/2205.13722},
      journal={The Fourth AAAI Workshop on Privacy-Preserving Artificial Intelligence (PPAI)},
      primaryClass={cs.LG}
}
```

Contribution is welcome. Do not hesitate to reach out via the issues or direct email to the operators.