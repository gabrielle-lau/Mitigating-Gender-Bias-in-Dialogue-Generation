# Mitigating Gender Bias in Dialogue Generation
This is the source code for my thesis on "Mitigating Gender Bias in Dialogue Generation" for the degree of MPhil in Machine Learning and Machine Intelligence at University of Cambridge.

The [ParlAI](https://github.com/facebookresearch/ParlAI) framework is utilised and the modified ParlAI scripts are included in `src/`. The modified code blocks are marked with the comment `# Thesis`.

An overview of the project is presented below.

## Introduction
Previous studies have found that these systems reflect or even amplify dataset biases. This thesis investigates how to mitigate gender bias in open-domain chatbots.  “Bias” is defined as “behaviour which systematically and unfairly discriminates against certain individuals or groups of individuals in favour of others” (Friedman and Nissenbaum, 1996). 

With the goal of reducing gender bias in open-domain chatbot models without compromising system response quality, this thesis explores two debiasing methods that do not require retraining a system from scratch:
* **Bias controlled finetuning** (Xu et al., 2020): performs continued training on a pre-trained model to learn to generate unbiased responses
* **Self-debiasing decoding** (Schick et al., 2021): uses a language model’s internal knowledge to algorithmically reduce biased responses generated during testing time

We apply the gender mitigation methods on a state-of-the-art open-domain dialogue model called [BlenderBot](https://ai.facebook.com/blog/state-of-the-art-open-source-chatbot/) (Roller et al., 2020), which is available on [ParlAI](https://github.com/facebookresearch/ParlAI). 

## Gender bias (& stereotype) controlled finetuning
### Approach
In order to finetune a pre-trained Reddit 90M model to generate an equal number of responses containing male words and female words respectively, Xu et al. (2020) control the system’s responses with bias control tokens of the form "f<sup>0/1</sup>m<sup>0/1</sup>" through conditional training. We followed this gender bias controlled finetuning approach for our baseline, but used a smaller 90M-parameter model due to limited computing resources. 

Before finetuning, each dialogue in the finetuning data is classified as 1 of 4 gender bias token classes depending on the presence of gendered words in the response:
1. **f0m0**:  no gender words.
2. **f0m1**:  at least one male word.
3. **f1m0**:  at least one female word.
4. **f1m1**:  at least one female word and at least one male word.

The appropriate token is appended to the context of each dialogue in the finetuning data, then we finetuned the model (batch size 32, Adamax optimiser, learning rate 8e-6) on 1 NVidia Tesla P100-PCIE-16GB GPU on HPC until convergence. The resulting model is called “GB-Ctrl”.

To extend Xu et al. (2020)'s approach, we  introduce  a  novel  set  of  tokens  as  control  variables  to  simultaneously  control  two types of gender bias – genderedness and stereotype. The tokens are of the form "f<sup>0/1</sup>m<sup>0/1</sup><sub>a/s/u</sub>", where a/s/u indicates if the response given the context is a(n):
1. **f<sup>0/1</sup>m<sup>0/1</sup>a**: anti-stereotype response.
2. **f<sup>0/1</sup>m<sup>0/1</sup>s**: stereotype response.
3. **f<sup>0/1</sup>m<sup>0/1</sup>u**: unrelated response.

We finetune a Reddit 90M model on finetuning data of Xu et al. (2020) with an additional dataset called “StereoSet” (Nadeem et al., 2020), with new tokens appended to the context in the data. The resulting model is called “GBS-Ctrl”

### Results
<img src="https://github.com/gabrielle-lau/Mitigating-Gender-Bias-in-Dialogue-Generation/blob/main/figures/images/GB-Ctrl-convai2-h.png" width="500">

<img src="https://github.com/gabrielle-lau/Mitigating-Gender-Bias-in-Dialogue-Generation/blob/main/figures/images/GBS-Ctrl-convai2-h.png" width="500">

<img src="https://github.com/gabrielle-lau/Mitigating-Gender-Bias-in-Dialogue-Generation/blob/main/figures/images/GBS-Ctrl-stereoset-h.png" width="500">
