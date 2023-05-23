Objective: Although pretrained Transformers such as BERT achieve high accuracy on indistribution examples, do they generalize to new distributions?

Introduction:
- The train and test distribution are often not identically distribued
- Models must generalize to OOD example whenever possible, and when OOD examples don't belong to any know class, models must detect them in order to abstain or trigger a conservative fallback policy
- Most evaluation in NLP assumes the train and test examples are independent and identically distributed (IID). In the IID setting, large pretrained Transformer models can attain near human-level performance on numerous tasks.
- Moreover, pretrained Transformers can rely heavily on spurious cues (虚假的线索)and annotation artifacts (不是本质 表面的东西) which OOD examples are likely to include, so their OOD robustness remains uncertain
- In this paper, we decompose OOD robustness into a model's ability to (1)generalize and to(2)detect OOD examples
  - To measure the OOD robustness, we create a new evaluation benchmark that tests robustness to shifts in writing style, topic, and vocabulary, and spans of several tasks. 
    - Using our OOD generalization benchmark, we show that pretrained Transformers are considerably more robust to OOD examples than traditional NLP models
    - Moreover, we demonstrate that while pretraining larger models does not seem to improve OOD generalization, pretraining models on diverse data does improve OOD generalization 
  - To measure the OOD detection performance, we turn classifiers into anomaly detectors by using their prediction confidences as anomaly scores
    - We show that many non-pretrained NLP models are often near or worse than random chance at OOD detection. In contrast, pretrained Transformers are far more capable at OOD detection 

How We Test Robustness:
- Train and Test Datasets: 
  - Each dataset either (1) contains metadata which allows us to naturally split the samples or (2) can be paired with a similar dataset from a distinct data generating process. By splitting or grouping our chosen datasets, we can induce a distribution shift and measure OOD generalization
    - train on one dataset and evaluate on the other dataset , and vice versa
    - Yelp review dataset contains restaurant reviews with detailed metadata. Carve out four groups from the dataset based on food type: American, Chinese, Italian, and Japanese

OOD Generalization:

OOD Detection:
- For evaluation, we follow past work, [Deep Anomaly Detection with Outlier Exposure](https://arxiv.org/abs/1812.04606), and report the False Alarm Rate of 95%Recall(FAR95). The FAR95 is the probability that an in-distribution example raises a false alarm, assuming that 95% of all out-of-distribution examples are detected. Hence a lower FAR95 is better

Discussion and Related Work:
- Why are pretrained models more robust? (mentioned many possibilies, didn't make any conclusion)
- Domain Adaptation, where models must learn representations of a source and target distribution $`\textcolor{red}{\text{(can read more papers about this)}}`$ 
- Counteracting 抵消 annotation artifacts: (not helpful solution, but mention about why) Annotators can accidentally leave unintended shortcuts
in datasets that allow models to achieve high accuracy by effectively “cheating”
