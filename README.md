# LLMs Interview 八股文


## 简介

本仓库为大模型面试相关概念，由本人参考网络资源整理，欢迎阅读，如果对你有用，麻烦点一下 `start`，谢谢！

为了在低资源情况下，学习大模型，进行动手实践，创建 [tiny-llm-zh](https://github.com/wdndev/tiny-llm-zh)仓库，旨在构建一个小参数量的中文Llama2大语言模型，方便学习，欢迎学习交流。

## 在线阅读

本仓库相关文章已放在个人博客中，欢迎阅读：

在线阅读链接：[LLMs Interview Note](http://wdndev.github.io/note/llm/llm_concept/llm%E5%85%AB%E8%82%A1.html)

## 注意：

相关答案为自己撰写，若有不合理地方，请指出修正，谢谢！

欢迎关注微信公众号，会不定期更新LLM内容，以及一些面试经验：

 <img src=https://github.com/wdndev/personal/blob/main/image/llmers_weixin.jpg width = "427" height = "156" alt="weixin" />


## 目录

* [首页](/)
* [真实面试题](/docs/ch1)
* [01.大语言模型基础](/docs/01.大语言模型基础/)
  * [1.1 大模型发展历程](/docs/01.大语言模型基础/)
    * [1.语言模型](/docs/01.大语言模型基础/1.语言模型/1.语言模型.md "1.语言模型")
  * [1.2 分词与词向量]()
    * [1.分词](/docs/01.大语言模型基础/1.分词/1.分词.md)
    * [2.jieba分词用法及原理](/docs/01.大语言模型基础/2.jieba分词用法及原理/2.jieba分词用法及原理.md)
    * [3.词性标注](/docs/01.大语言模型基础/3.词性标注/3.词性标注.md)
    * [4.句法分析](/docs/01.大语言模型基础/4.句法分析/4.句法分析.md "4.句法分析")
    * [5.词向量](/docs/01.大语言模型基础/5.词向量/5.词向量.md "5.词向量")
  * [1.3 语言模型基础知识](/docs/01.大语言模型基础/)
    * [Word2Vec](/docs/01.大语言模型基础/Word2Vec/Word2Vec.md "Word2Vec")
    * [NLP三大特征抽取器（CNN/RNN/TF）](/docs/01.大语言模型基础/NLP三大特征抽取器（CNN-RNN-TF）/NLP三大特征抽取器（CNN-RNN-TF）.md)
    * [NLP面试题](/docs/01.大语言模型基础/NLP面试题/NLP面试题.md "NLP面试题")
    * [LLM为什么Decoder only架构](</docs/01.大语言模型基础/LLM为什么Decoder only架构/LLM为什么Decoder only架构.md> "LLM为什么Decoder only架构")
  * [1.4 深度学习](/docs/01.大语言模型基础/)
    * [1.激活函数](/docs/01.大语言模型基础/1.激活函数/1.激活函数.md)
  * [1.5 一些题目](/docs/01.大语言模型基础/)
    * [1.llm概念](/docs/01.大语言模型基础/1.llm概念/1.llm概念.md)
* [02.大语言模型架构](/docs/02.大语言模型架构/)
  * [2.1 Transformer模型](/docs/02.大语言模型架构/)
    * [1.attention](/docs/02.大语言模型架构/1.attention/1.attention.md "1.attention")
    * [2.layer\_normalization](/docs/02.大语言模型架构/2.layer_normalization/2.layer_normalization.md "2.layer_normalization")
    * [3.位置编码](/docs/02.大语言模型架构/3.位置编码/3.位置编码.md "3.位置编码")
    * [4.tokenize分词](/docs/02.大语言模型架构/4.tokenize分词/4.tokenize分词.md "4.tokenize分词")
    * [5.token及模型参数](/docs/02.大语言模型架构/5.token及模型参数/5.token及模型参数.md "5.token及模型参数")
    * [6.激活函数](/docs/02.大语言模型架构/6.激活函数/6.激活函数.md "6.激活函数")
  * [2.2 注意力](/docs/02.大语言模型架构/)
    * [MHA\_MQA\_GQA](/docs/02.大语言模型架构/MHA_MQA_GQA/MHA_MQA_GQA.md "MHA_MQA_GQA")
  * [2.3 解码部分](/docs/02.大语言模型架构/)
    * [解码策略（Top-k & Top-p & Temperature）](</docs/02.大语言模型架构/解码策略（Top-k & Top-p & Temperatu/解码策略（Top-k & Top-p & Temperature）.md> "解码策略（Top-k & Top-p & Temperature）")
  * [2.4 BERT](/docs/02.大语言模型架构/)
    * [bert细节](/docs/02.大语言模型架构/bert细节/bert细节.md "bert细节")
    * [Transformer架构细节](/docs/02.大语言模型架构/Transformer架构细节/Transformer架构细节.md "Transformer架构细节")
    * [bert变种](/docs/02.大语言模型架构/bert变种/bert变种.md "bert变种")
  * [2.5 常见大模型](/docs/02.大语言模型架构/)
    * [llama系列模型](/docs/02.大语言模型架构/llama系列模型/llama系列模型.md "llama系列模型")
    * [chatglm系列模型](/docs/02.大语言模型架构/chatglm系列模型/chatglm系列模型.md "chatglm系列模型")
    * [llama 2代码详解](</docs/02.大语言模型架构/llama 2代码详解/llama 2代码详解.md> "llama 2代码详解")
    * [llama 3](</docs/02.大语言模型架构/llama 3/llama 3.md> "llama 3")
  * [2.6 MoE](/docs/02.大语言模型架构/)
    * [1.MoE论文](/docs/02.大语言模型架构/1.MoE论文/1.MoE论文.md "1.MoE论文")
    * [2.MoE经典论文简牍](/docs/02.大语言模型架构/2.MoE经典论文简牍/2.MoE经典论文简牍.md "2.MoE经典论文简牍")
    * [3.LLM MoE ：Switch Transformers](</docs/02.大语言模型架构/3.LLM MoE ：Switch Transformers/3.LLM MoE ：Switch Transformers.md> "3.LLM MoE ：Switch Transformers")
* [03.训练数据集](/docs/03.训练数据集/)
  * [3.1 数据集](/docs/03.训练数据集/)
    * [数据格式](/docs/03.训练数据集/数据格式/数据格式.md "数据格式")
  * [3.2 模型参数](/docs/03.训练数据集/)
* [04.分布式训练](/docs/04.分布式训练/)
  * [4.1 基础知识](/docs/04.分布式训练/)
    * [1.概述](/docs/04.分布式训练/1.概述/1.概述.md "1.概述")
    * [2.数据并行](/docs/04.分布式训练/2.数据并行/2.数据并行.md "2.数据并行")
    * [3.流水线并行](/docs/04.分布式训练/3.流水线并行/3.流水线并行.md "3.流水线并行")
    * [4.张量并行](/docs/04.分布式训练/4.张量并行/4.张量并行.md "4.张量并行")
    * [5.序列并行](/docs/04.分布式训练/5.序列并行/5.序列并行.md "5.序列并行")
    * [6.多维度混合并行](/docs/04.分布式训练/6.多维度混合并行/6.多维度混合并行.md "6.多维度混合并行")
    * [7.自动并行](/docs/04.分布式训练/7.自动并行/7.自动并行.md "7.自动并行")
    * [8.moe并行](/docs/04.分布式训练/8.moe并行/8.moe并行.md "8.moe并行")
    * [9.总结](/docs/04.分布式训练/9.总结/9.总结.md "9.总结")
  * [4.2 DeepSpeed](/docs/04.分布式训练/)
    * [deepspeed介绍](/docs/04.分布式训练/deepspeed介绍/deepspeed介绍.md "deepspeed介绍")
  * [4.3 Megatron](/docs/04.分布式训练/)
  * [4.4 训练加速](/docs/04.分布式训练/)
  * [4.5 一些有用的文章](/docs/04.分布式训练/)
  * [4.6 一些题目](/docs/04.分布式训练/)
    * [1.分布式训练题目](/docs/04.分布式训练/分布式训练题目/分布式训练题目.md "分布式训练题目")
    * [2.显存问题](/docs/04.分布式训练/1.显存问题/1.显存问题.md "1.显存问题")
* [05.有监督微调](/docs/05.有监督微调/)
  * [5.1 理论](/docs/05.有监督微调/)
    * [1.基本概念](/docs/05.有监督微调/1.基本概念/1.基本概念.md "1.基本概念")
    * [2.prompting](/docs/05.有监督微调/2.prompting/2.prompting.md "2.prompting")
    * [3.adapter-tuning](/docs/05.有监督微调/3.adapter-tuning/3.adapter-tuning.md "3.adapter-tuning")
    * [4.lora](/docs/05.有监督微调/4.lora/4.lora.md "4.lora")
    * [5.总结](/docs/05.有监督微调/5.总结/5.总结.md "5.总结")
  * [5.2 微调实战](/docs/05.有监督微调/)
    * [llama2微调](/docs/05.有监督微调/llama2微调/llama2微调.md "llama2微调")
    * [ChatGLM3微调](/docs/05.有监督微调/ChatGLM3微调/ChatGLM3微调.md "ChatGLM3微调")
  * [5.3 一些题目](/docs/05.有监督微调/)
    * [1.微调](/docs/05.有监督微调/1.微调/1.微调.md "1.微调")
    * [2.预训练](/docs/05.有监督微调/2.预训练/2.预训练.md "2.预训练")
* [06.推理](/docs/06.推理/)
  * [6.1 推理框架](/docs/06.推理/)
    * [0.llm推理框架简单总结](/docs/06.推理/0.llm推理框架简单总结/0.llm推理框架简单总结.md "0.llm推理框架简单总结")
    * [1.vllm](/docs/06.推理/1.vllm/1.vllm.md "1.vllm")
    * [2.text_generation\_inference](/docs/06.推理/2.text_generation_inference/2.text_generation_inference.md "2.text_generation_inference")
    * [3.faster_transformer](/docs/06.推理/3.faster_transformer/3.faster_transformer.md "3.faster_transformer")
    * [4.trt_llm](/docs/06.推理/4.trt_llm/4.trt_llm.md "4.trt_llm")
  * [6.2 推理优化技术](/docs/06.推理/)
    * [llm推理优化技术](/docs/06.推理/llm推理优化技术/llm推理优化技术.md "llm推理优化技术")
  * [6.3 量化](/docs/06.推理/)
  * [6.4 vLLM](/docs/06.推理/)
  * [6.5 一些题目](/docs/06.推理/)
    * [1.推理](/docs/06.推理/1.推理/1.推理.md "1.推理")
* [07.强化学习](/docs/07.强化学习)
  * [7.1 强化学习原理](/docs/07.强化学习)
    * [策略梯度（pg）](/docs/07.强化学习/策略梯度（pg）/策略梯度（pg）.md "策略梯度（pg）")
    * [近端策略优化(ppo)](/docs/07.强化学习/近端策略优化(ppo)/近端策略优化(ppo).md "近端策略优化(ppo)")
  * [7.2 RLHF](/docs/07.强化学习)
    * [大模型RLHF：PPO原理与源码解读](/docs/07.强化学习/大模型RLHF：PPO原理与源码解读/大模型RLHF：PPO原理与源码解读.md "大模型RLHF：PPO原理与源码解读")
    * [DPO](/docs/07.强化学习/DPO/DPO.md "DPO")
  * [7.3 一些题目](/docs/07.强化学习)
    * [1.rlhf相关](/docs/07.强化学习/1.rlhf相关/1.rlhf相关.md "1.rlhf相关")
    * [2.强化学习](/docs/07.强化学习/2.强化学习/2.强化学习.md "2.强化学习")
* [08.检索增强RAG](/docs/08.检索增强rag/)
  * [8.1 RAG](/docs/08.检索增强rag/)
    * [检索增强llm](/docs/08.检索增强rag/检索增强llm/检索增强llm.md "检索增强llm")
    * [rag（检索增强生成）技术](/docs/08.检索增强rag/rag（检索增强生成）技术/rag（检索增强生成）技术.md "rag（检索增强生成）技术")
  * [8.2 Agent](/docs/08.检索增强rag/)
    * [大模型agent技术](/docs/08.检索增强rag/大模型agent技术/大模型agent技术.md "大模型agent技术")
* [09.大语言模型评估](/docs/09.大语言模型评估/)
  * [9.1 模型评估](/docs/09.大语言模型评估/)
    * [1.评测](/docs/09.大语言模型评估/1.评测/1.评测.md "1.评测")
  * [9.2 LLM幻觉](/docs/09.大语言模型评估/)
    * [1.大模型幻觉](/docs/09.大语言模型评估/1.大模型幻觉/1.大模型幻觉.md "1.大模型幻觉")
    * [2.幻觉来源与缓解](/docs/09.大语言模型评估/2.幻觉来源与缓解/2.幻觉来源与缓解.md "2.幻觉来源与缓解")
* [10.大语言模型应用](/docs/10.大语言模型应用/)
  * [10.1 思维链提示](/docs/10.大语言模型应用/)
    * [1.思维链（cot）](/docs/10.大语言模型应用/1.思维链（cot）/1.思维链（cot）.md "1.思维链（cot）")
  * [10.2 LangChain框架](/docs/10.大语言模型应用/)
    * [1.langchain](/docs/10.大语言模型应用/1.langchain/1.langchain.md "1.langchain")
* [98.相关课程](/docs/98.相关课程/)
* [99.参考资料](/docs/99.参考资料/)

## 更新记录

- 2024.05.04 ： 使用 docsify 构建在线版本
- 2024.05.01 : 解码参数，策略
- 2024.04.15 : BERT细节
- 2024.03.19 : 推理参数
- 2024.03.17 ： 强化学习部分，PG，PPO，RLHF，DPO
- 2024.03.13 ： 强化学习题目
- 2024.03.10 : LLMs相关课程
- 2024.03.08 ： RAG技术
- 2024.03.05 ：大模型评估，幻觉
- 2024.01.26 ：语言模型简介
- 2023.12.15 ： llama，chatglm 架构
- 2023.12.02 ：LLM推理优化技术
- 2023.12.01 ：调整目录
- 2023.11.30 ：18.Layer-Normalization，21.Attention升级
- 2023.11.29 ： 19.激活函数，22.幻觉，23.思维链
- 2023.11.28 ： 17.位置编码
- 2023.11.27 ： 15.token及模型参数， 16.tokenize分词
- 2023.11.25 ： 13.分布式训练
- 2023.11.23 ： 6.推理， 7.预训练， 8.评测，9.强化学习， 11.训练数据集，12.显存问题,14.agent
- 2023.11.22 ： 5.高效微调
- 2023.11.10 ： 4.LangChain
- 2023.11.08 ： 建立仓库；1.基础，2.进阶，3.微调





