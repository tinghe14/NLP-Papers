# NLP-Papers
This is a repository for organizing articles related to NLP. Most papers are linked to my reading notes. Feel free to contact me for discussion via Linkedin.

markdown syntax shortcut:
- highlighter:
$`\textcolor{red}{\text{1}}`$ 
$`\textcolor{blue}{\text{2}}`$ 
$`\textcolor{green}{\text{3}}`$
- shortcut:
<a id='tag'></a> (#tag)

# Table of Contents (ongoing)
1. [large language model](#llm)
2. [acceleration NLP model](#accelerate)
4. [transformer](#transformer)
5. [embedding](#embedding)

# Large Language Model
<a id='llm'></a>
## 2023
1. Paper - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971.pdf):open-sourced LLM for research community. The most benfit point to me is that I can now follow how researcher use LLaMA to fine tune specific task
    - [note](https://docs.google.com/presentation/d/1TLGVurmYcE_nqks2V1-i1n5Jnj2Z-AzZ6sQnqqiQ3gA/edit?usp=sharing)
<!--
4. [Self-Instruct]()
5. [Standford Alpaca]()
6. [SLiC-HF: Sequence Likelihood Calibration with Human Feedback]():replace RL techniques
7. [Stanford Webinar-GPT-3 & Beyond](https://www.youtube.com/results?search_query=stanford+webinar+-+gpt-3+%26+beyond)
8. 8. Alignment instead of RL
-->
## 2020
1. Paper - [Lanugage Models are Few-shot Learners](https://arxiv.org/abs/2005.14165): the performance of fine-tuned models on specific benchmarks, even when it is nominally at human-level, may exaggerate actual performance on the underlying task <a id='lmsfewshortlearner'></a>
    - [note](https://github.com/tinghe14/NLP-Papers/blob/ba4b2784f280fbe784de215651e51592367e8bed/large%20language%20model/2_LM_few_shot_learners/2%20note.md)
## Other
1. Cousera - DeepLearning.AI ChatGPT Prompt Enginnering for Developers:
    - [note-ongoing]()
2. Blog in Chinese - [Introduction to ChatGPT/ChatGPT 基础科普：知其一点所以然](https://yam.gift/2023/04/15/NLP/2023-04-15-ChatGPT-Introduction/): introduce RNN to GPT series and share about RLHF
    - [note](https://github.com/tinghe14/NLP-Papers/blob/506df334b52d332b682b5bbf1c402119c8c57d3b/large%20language%20model/0%20note_ChatGPT%E5%9F%BA%E7%A1%80%E7%A7%91%E6%99%AE.md)
3. Blog in Chinese- [ChatGPT 使用指南：相似匹配](https://github.com/datawhalechina/hugging-llm/blob/main/content/ChatGPT%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97%E2%80%94%E2%80%94%E7%9B%B8%E4%BC%BC%E5%8C%B9%E9%85%8D.ipynb):
    - [note - ongoing]()
<!--
2. Video in Chinese - [GPT，GPT-2，GPT-3 论文精读【论文精读】](https://www.bilibili.com/video/BV1AF411b7xQ/?spm_id_from=333.999.0.0&vd_source=8b4794944ae27d265c752edb598636de)
3. Video in Chinese - [InstructGPT 论文精读【论文精读·48】](https://www.bilibili.com/video/BV1hd4y187CR/?spm_id_from=333.999.0.0&vd_source=8b4794944ae27d265c752edb598636de)
4. Video in Chinese- [GPT-4论文精读【论文精读·53】](https://www.bilibili.com/video/BV1vM4y1U7b5/?spm_id_from=333.999.0.0&vd_source=8b4794944ae27d265c752edb598636de)
-->

# Finer Topics
## Detect Out-of-Distribution
### 2020
1. Paper - (ACL main)[Pretrained transformers improve out-of-distribution robustness](https://arxiv.org/pdf/2004.06100.pdf): cited by [LMs are few-shot learners](#lmsfewshortlearner), but I think this is not a good paper to support their idea in GPT3. Whatever, the trick of turing classifier into anomaly detector to help to measure preformance of OOD generalization can help my work. $`\textcolor{red}{\text{how to calculate recall95? offline model or online inference can use? want to look at one paper about human alignment?LIMA}}`$ 
    - [my note](https://github.com/tinghe14/NLP-Papers/blob/4daf9f39aa31da7e1bcade58166c389912eef1c5/transformer/0_pretrain_transformers_improve_ood/note.md)/[code](https://github.com/camelop/NLP-Robustness)
## Data Augmentation

# Acceleration NLP Models
## Other
1. Video in Chinese - [AI框架-分布式并行及其策略](https://www.bilibili.com/video/BV1ge411L7mi/?spm_id_from=333.788&vd_source=8b4794944ae27d265c752edb598636de): introduce to distributed cluster, computer network and training large AI models https://space.bilibili.com/517221395/channel/collectiondetail?sid=936465
    - [note-ongoing]()
2. Video in Chinese - [AI编译器-传统编译器](https://space.bilibili.com/517221395/channel/collectiondetail?sid=857162) :introduce to traditional compiler vs AI compiler and optimized operator https://space.bilibili.com/517221395/channel/collectiondetail?sid=907218 https://space.bilibili.com/517221395/channel/collectiondetail?sid=960739 https://space.bilibili.com/517221395/channel/collectiondetail?sid=933181
    - [note-ongoing]()
3. Video in Chinese - [推理系统-整体概述]（https://space.bilibili.com/517221395/channel/collectiondetail?sid=997962）
    - [note-ongoing]()
4. Video in Chinese - [AI编译器-pytorch编译优化]（https://space.bilibili.com/517221395/channel/collectiondetail?sid=914908）
    - [note-going]()
5. Video in Chinese - [推理引擎-轻量网络]（https://space.bilibili.com/517221395/channel/collectiondetail?sid=1018326）
    - [note-going]()
6. Video in Chinese - [推理引擎-模型压缩]（https://space.bilibili.com/517221395/channel/collectiondetail?sid=1038745）
    - [note-going]()
7. Video in Chinese - [推理引擎-离线转换&优化]（https://space.bilibili.com/517221395/channel/collectiondetail?sid=1055386）
    - [note-going]()
8. Video in Chinese - [AI芯片-计算体系，芯片基础，gpu原理]https://space.bilibili.com/517221395/channel/series
9. Video - [DeepSpeed: All the tricks to scale to gigantic models](https://www.youtube.com/watch?v=pDGI668pNg0)
10. Documentation - [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)
    - [note]()
12. Tutorial in Chinese - [Pytorch Lightning 完全攻略](https://zhuanlan.zhihu.com/p/353985363)
    - [note]()
13. Video in Chinese - [A great pytorch extension: pytorch-lightning lets you double your coding efficiency! ](https://www.youtube.com/watch?v=O7dNXpgdWbo)
    - [note]()
<a id='accelerate'></a>
<!--
## 2023
1. Blog in Chinese - [Google新作试图“复活”RNN：RNN能否再次辉煌？](https://spaces.ac.cn/archives/9554)
2. Blog in Chinese - [Google新搜出的优化器Lion：效率与效果兼得的“训练狮”](https://spaces.ac.cn/archives/9473)
## 2022
1. Paper - [A Survey on Model Compression and Acceleration for Pretrained Language Models]
2. Blog - [How Cohere is accelerating language model training with Google Cloud TPUs]
3. Blog in Chinese - [基于Amos优化器思想推导出来的一些“炼丹策略”](https://spaces.ac.cn/archives/9344)
## 2021
1. Blog - [Hugging Face blog: How we sped up transformer inference 100x for Hugging Face API customers](https://huggingface.co/blog/accelerated-inference)
## Other
1. Tutorial - [Transformers Math 101](https://eleutherai.notion.site/Transformers-Math-101-d2fcfc7a25d446388fde97821ad2412a)
2. PyTorch Documentaiton - [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
3. Blog - 并行计算入门 不知道哪个好 要先挑选下：-给自己发的微信
    - https://medium.com/nlplanet/two-minutes-nlp-leveraging-parallelisms-to-train-large-neural-networks-a7f31de06eac
    - https://zhuanlan.zhihu.com/p/157884112 
    - http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/AI%20%E9%83%A8%E7%BD%B2%E5%8F%8A%E5%85%B6%E5%AE%83%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/%E5%B9%B6%E5%8F%91%E7%BC%96%E7%A8%8B/
4. Blog in Chinese - [一文总结当下常用的大型 transformer 效率优化方案](https://zhuanlan.zhihu.com/p/623744798)
    - 类似的 http://121.199.45.168:8234/7/
6. Tool - [LightSeq, 支持Transformer全流程训练加速，最高加速3倍！字节跳动LightSeq上新)](https://www.jiqizhixin.com/articles/2021-06-24-13)
    - 类似的 http://giantpandacv.com/academic/%E7%AE%97%E6%B3%95%E7%A7%91%E6%99%AE/Transformer/LightSeq%20Transformer%E9%AB%98%E6%80%A7%E8%83%BD%E5%8A%A0%E9%80%9F%E5%BA%93/
8. Tool - DeepSpeed
    - https://blog.csdn.net/CheatEngine_jaz/article/details/124629041
    - 好像只支持huggingface和pytorch lighting 不知道是否支持pytorch https://www.deepspeed.ai/getting-started/
    - 好像很麻烦 配置一堆问题 小红书群里说的
10. GCP Documentation - [Cloud TPU性能指南]（https://cloud.google.com/tpu/docs/cloud-tpu-tools?hl=zh-cn）
-->

# Transformer 
<a id='transformer'></a>
## 2020
1. 
## 2019
1. Paper - [Deep Anomaly Detection with Outlier Exposure](https://arxiv.org/pdf/1812.04606.pdf)
    - [note - ongoing]()
## 2018
1. Paper - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
    - [note](https://medium.com/@hetinghelen/tasks-and-common-models-in-natural-language-processing-11c523d88f02)
## 2017
1. Paper - [Attention is all your need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf): Milestone of Transformer architecture
    - [note](https://medium.com/@hetinghelen/tasks-and-common-models-in-natural-language-processing-11c523d88f02)

# Embedding
<a id='embedding'></a>
## 2015
1. Paper - [Improving Distribution Similarity with Lessons Learned from Word Embeddings](https://aclanthology.org/Q15-1016/): evaluation in word embedding
    - [note](https://medium.com/@hetinghelen/tasks-and-common-models-in-natural-language-processing-11c523d88f02)
2. Paper - [Evaluation methods for unsupervised word embedding](https://aclanthology.org/D15-1036/)
    - [note](https://medium.com/@hetinghelen/tasks-and-common-models-in-natural-language-processing-11c523d88f02)
## 2014
1. Paper - [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
    - [note](https://medium.com/@hetinghelen/tasks-and-common-models-in-natural-language-processing-11c523d88f02)
## 2013
1. Paper - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
    - [note](https://medium.com/@hetinghelen/tasks-and-common-models-in-natural-language-processing-11c523d88f02)
2. Paper - [Distributed Representations of Words and Phrases and their Compositionality](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf): negative sampling techniques
    - [note](https://medium.com/@hetinghelen/tasks-and-common-models-in-natural-language-processing-11c523d88f02)


# NER

# RE
