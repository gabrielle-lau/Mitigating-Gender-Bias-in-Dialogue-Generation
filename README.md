# Mitigating Gender Bias in Dialogue Generation
This is the source code for my thesis on "Mitigating Gender Bias in Dialogue Generation" submitted for the MPhil in Machine Learning and Machine Intelligence at University of Cambridge.

The [ParlAI](https://github.com/facebookresearch/ParlAI) Python framework (version 1.2.0) is utilised throughout the thesis. The modified ParlAI scripts are included in `src/`, with the modified code blocks indicated with the comment `# Thesis`. The other scripts (primarily written using standard Python packages) are incluced in `scripts/`.

An overview of the project is presented below.

## Introduction
Previous studies have found that these systems reflect or even amplify dataset biases. This thesis investigates how to mitigate gender bias in open-domain chatbots.  “Bias” is defined as “behaviour which systematically and unfairly discriminates against certain individuals or groups of individuals in favour of others” (Friedman and Nissenbaum, 1996). 

With the goal of reducing gender bias in open-domain chatbot models without compromising system response quality, this thesis explores two debiasing methods that do not require retraining a system from scratch:
* **Bias controlled finetuning** (Xu et al., 2020): performs continued training on a pre-trained model to learn to generate unbiased responses
* **Self-debiasing decoding** (Schick et al., 2021): uses a language model’s internal knowledge to algorithmically reduce biased responses generated during testing time

We apply the gender mitigation methods on a state-of-the-art open-domain dialogue model called [BlenderBot](https://ai.facebook.com/blog/state-of-the-art-open-source-chatbot/) (Roller et al., 2020), which is available on [ParlAI](https://github.com/facebookresearch/ParlAI). Blender has a Seq2Seq transformer-based encoder-decoder architecture.

## Gender bias (& stereotype) controlled finetuning
### Approach
In order to finetune a pre-trained Reddit 90M model to generate an equal number of responses containing male words and female words respectively, Xu et al. (2020) control the system’s responses with bias control tokens of the through conditional training. We followed this gender bias controlled finetuning approach for our baseline, but used a smaller 90M-parameter model due to limited computing resources. 

Before finetuning, each dialogue in the finetuning data (ConvAI2, Empathetic Dialogues, Wizard of Wikipedia, Blended Skill Talk) is classified as 1 of 4 gender bias token classes depending on the presence of gendered words in the response:
1. **f0m0**:  no gender words.
2. **f0m1**:  at least one male word.
3. **f1m0**:  at least one female word.
4. **f1m1**:  at least one female word and at least one male word.

The appropriate token is appended to the context of each dialogue in the finetuning data, then we finetuned the model (batch size 32, Adamax optimiser, learning rate 8e-6) on 1 NVidia Tesla P100-PCIE-16GB GPU on HPC until convergence. The resulting model is called “GB-Ctrl”.

To extend Xu et al. (2020)'s approach, we  introduce  a  novel  set  of  tokens  as  control  variables  to  simultaneously  control  two types of gender bias – genderedness and stereotype. The 12 tokens are of the form "f<sup>0/1</sup>m<sup>0/1</sup><sub>a/s/u</sub>", where a/s/u indicates if the response given the context is a(n) anti-stereotype, stereotype or unrelated response. 3 of 12 tokens are listed below:
1. **f0m0a**: genderless and anti-stereotype.
2. **f0m0s**: genderless and stereotype.
3. **f0m0u**: genderless and unrelated.

We finetuned a Reddit 90M model on the above-mentioned finetuning data and an additional dataset called “StereoSet” (Nadeem et al., 2020), with new tokens appended to the context in the data. The resulting model is called “GBS-Ctrl”

### Evaluation metrics
We evaluated GB(S)-Ctrl on ConvAI2 and StereoSet using the following metrics:
* **Total toxicity**: the percentage of model responses flagged as offensive by a string matcher or safety classifier.
* **Female (Male)%**: the percentage of responses containing at least one female (male) word.
* **% Delta stereotype bias score**: a smaller delta means a more equal percentage of positive and negative bias scores, thus the less biased the model is to either stereotype or anti-stereotype.  Delta equals 0 is the ideal result.  A positive delta means the model is more likely to produce a stereotyped response, and vice versa.
* **Perplexity**: the exponentiated average negative log-likelihood of a tokenized sequence, a measure of dialogue fluency.

### Results on genderedness mitigation
* This gender bias (& stereotype) controlled finetuning approach is effective in reducing genderedness (ie. Female (Male)%) and toxicity, without worsening perplexity.
* In Figure 1, the “f0m0” genderless token is the best token for GB-Ctrl to conditionally generate less gendered model responses.  Genderedness decreases by more than half of that of Blender 90M. 
* In Figure 2-3, GBS-Ctrl with “f0m0u” is about as effective  as  GB-Ctrl  with  “f0m0”  for  reducing  genderedness  on  ConvAI2  and StereoSet.

#### Figure 1: Results of GB-Ctrl evaluated on ConvAI2 validation set
<img src="https://github.com/gabrielle-lau/Mitigating-Gender-Bias-in-Dialogue-Generation/blob/main/figures/images/GB-Ctrl-convai2-h.png" width="500">

#### Figure 2: Results of GBS-Ctrl evaluated on ConvAI2 validation set
<img src="https://github.com/gabrielle-lau/Mitigating-Gender-Bias-in-Dialogue-Generation/blob/main/figures/images/GBS-Ctrl-convai2-h.png" width="500">

#### Figure 3: Results of GBS-Ctrl evaluated on StereoSet validation set
<img src="https://github.com/gabrielle-lau/Mitigating-Gender-Bias-in-Dialogue-Generation/blob/main/figures/images/GBS-Ctrl-stereoset-h.png" width="500">

### Results on stereotype bias mitigation
* In Figure 4, GBS-Ctrl appears to reduce gender stereotype bias slightly, because the %delta of -31% is 2% smaller in magnitude than that of GB-Ctrl.  GB-Ctrlhas  -33%  %delta,  meaning  there  are  more  anti-stereotype  biases  than  stereotype biases.
* However, our gender bias & stereotype controlled finetuning approach has limitations in evaluating stereotype bias due to its reliance on StereoSet.

#### Figure 4: %Delta between positive and negative gender stereotype bias scores on StereoSet
<img src="https://github.com/gabrielle-lau/Mitigating-Gender-Bias-in-Dialogue-Generation/blob/main/figures/images/delta_gender_bias.png" width="500">

#### Figure 5: %Delta between positive and negative stereotype bias scores on StereoSet
<img src="https://github.com/gabrielle-lau/Mitigating-Gender-Bias-in-Dialogue-Generation/blob/main/figures/images/delta_bias.png" width="500">

### Results on classification accuracy
* GB(S)-Ctrl  reduced  classification  error  significantly.   This  indicates  both  models are finetuned as intended and have learned correct associations between token and target responses.
* GB-Ctrl model has one-third fewer errors than a 4-class random  classifier (48%  compared  to  75%),  and  the  GBS-Ctrl  almost  halved  the12-class random classifier’s error rate from 91% to 47%.
* GB(S)-Ctrl roughly halved the f0m0(u)-always classi-fier’s error rate from 14% to 7%.
* Figure 6(a), 7(a) and 8(a) show a dark blue diagonal, which means there is a high true positive rate for token classification.

#### Figure 6: Normalised confusion matrices of GB-Ctrl token classification on ConvAI2
(a) given  a  random  incorrect  token |  (b) given  a  fixed  f0m0  token
:-------------------------:|:-------------------------:
<img src="https://github.com/gabrielle-lau/Mitigating-Gender-Bias-in-Dialogue-Generation/blob/main/figures/images/GB-Ctrl-random-convai2.png" width="300">  |  <img src="https://github.com/gabrielle-lau/Mitigating-Gender-Bias-in-Dialogue-Generation/blob/main/figures/images/GB-Ctrl-f0m0-convai2.png" width="300">

#### Figure 6: Normalised confusion matrices of GBS-Ctrl token classification on ConvAI2
(a) given  a  random  incorrect  token |  (b) given  a  fixed  f0m0u  token
:-------------------------:|:-------------------------:
<img src="https://github.com/gabrielle-lau/Mitigating-Gender-Bias-in-Dialogue-Generation/blob/main/figures/images/GBS-Ctrl-random-convai2.png" width="300">  |  <img src="https://github.com/gabrielle-lau/Mitigating-Gender-Bias-in-Dialogue-Generation/blob/main/figures/images/GBS-Ctrl-f0m0u-convai2.png" width="300">

#### Figure 7: Normalised confusion matrices of GBS-Ctrl token classification on StereoSet
(a) given  a  random  incorrect  token |  (b) given  a  fixed  f0m0u  token
:-------------------------:|:-------------------------:
<img src="https://github.com/gabrielle-lau/Mitigating-Gender-Bias-in-Dialogue-Generation/blob/main/figures/images/GBS-Ctrl-random-stereoset.png" width="300">  |  <img src="https://github.com/gabrielle-lau/Mitigating-Gender-Bias-in-Dialogue-Generation/blob/main/figures/images/GBS-Ctrl-f0m0u-stereoset.png" width="300">

## Self-debiasing decoding
### Approach
Introduced by Schick et al. (2021),  self-debiasing is defined as a language model using only its internal knowledge to adapt its generation process to reduce the probability of generating texts that exhibit undesired behaviours. The principle concept uses zero-shot learning with textual bias descriptions,  where the system identifies and avoids specific biases. 

We extend Schick et al. (2021)'s approach to debias hostile sexism in GB-Ctrl, GBS-Ctrl and Blender 90M system by solving 3 key problems:
1.  Debias hostile sexism instead of toxicity.
2.  Blender is a dialogue model that cannot continue an incomplete sentence, but canonly reply to a complete sentence.
3.  The evaluation method in the paper (Schick et al., 2021) relies on [Perspective API](https://www.perspectiveapi.com/) for detecting biases, so it cannot easily test the effectiveness of the self-debiasing decoding algorithm in debiasing gender bias that is not measured by PerspectiveAPI.

We implemented the algorithm and designed novel dialogue templates to ask the system if a sexist tweet from a hostile sexism dataset (Waseem and Hovy, 2016; Jha and Mamidi, 2017) is acceptable, so  that  the  system’s  yes-no  answer  can  be  classified  as  “agree”,  “disagree”  or “neither agree nor disagree” to indicate if it contains hostile sexism. A response that says  yes  or  agrees  is  sexist,  since  it  is  a  harmful  affirmation  of  a  sexist  statement. In contrast, a system response that says no or disagrees is not sexist, since the response is a counter-speech to hate speech. 

We finetuned a RoBERTa base model (Liu  et  al.,  2019) on the Situations With Adversarial Generations (SWAG) dataset (Zellers et al., 2018) to automatically evaluate system responses on hostile sexism. The accuracy on the sexist task is around 70% based on human judgement, which is more than double of a 3-class random classifier. We call this model "RoBERTa MC".

### Evaluation metric of hostile sexism
We measure the percentage of system responses classified by RoBERTa MC as "agree", "disagree" or "neither agree nor disagree", before and after applying self-debiasing decoding. 

A reduction in hostile sexism is indicated by either:
* A decrease in the percentage of "agree", or 
* An increase in the percentage of "disagree".

### Results on hostile sexism mitigation
* From Table 1 Rows 1-2, self-debiasing  decoding (with hyperparameter λ=50) effectively  reduced  hostile  sexism  in  GBS-Ctrl and GB-Ctrl responses by 13% and 20% respectively compared to no self-debiasing.
* There is no negative effect on perplexity.

#### Table 1: Percentage change in classification with self-debiasing (λ=50) compared to without self-debiasing 
Row # | Model |  Agree | Disagree | Neither
:----:|:------:|:------:|:------:|:------:
1 | GB-Ctrl | -20.23% | +16.03% | +15.50%
2 | GBS-Ctrl | -13.03% | +13.32% | +17.92%
3 | Blender 90M | +2.91% | +6.87% | -6.53%

## Conclusion and contributions
The contributions made by this thesis are highlighted here:
* Developed a bias controlled finetuning approach that extends the approach in literature to simultaneously reduce gendered words and stereotype bias in a state-of-the-art open-domain chatbot, by introducing novel bias control variables. 
* Extended literature's self-debiasing decoding algorithm to debias hostile sexism in dialogue systems.
* Introduced a novel, general  approach to evaluate hostile sexism in dialogue system responses using  RoBERTa for classifying harmful affirmation.
* Combined these two finetuning and decoding approaches to mitigate multiple types of gender biases. 
