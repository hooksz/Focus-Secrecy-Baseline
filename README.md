# Can Foundation Models Help Us Achieve Perfect Secrecy? A simple baseline for personal & private ML.

This repository explores the baseline of using in-context learning for personal & private machine learning! We include scripts to download data and evaluate foundation models of various sizes and types across popular personal machine learning benchmarks from the privacy literature. Additional information can be found in the paper: https://arxiv.org/abs/2205.13722. Contributions of additional benchmarks and baselines are welcome in the form of PRs to the benchmarks folder and additions of a data loader. Exciting future questions: 

- Do we see further personalization with more in-context examples and longer contexts? 
- Can we understand the limits of this baseline in a more principled manner? Are the tasks we've evaluated on so far too similar to the pretraining distribution? [Maybe we should construct new privacy benchmarks!](https://arxiv.org/abs/2212.06470)
- Can we enable better in-context learning quality in smaller & open-source models? (Checkout recent work: [Ask Me Anything](https://arxiv.org/abs/2210.02441))
- Other ways of using the FMs -- is it better to generate synthetic data with the FMs and then train locally?

<p align="center"><img width="85%" src="imgs/main_figure.png" /></p>

## Setup

Use the following commands to clone and install this package. We highly recommend you use con