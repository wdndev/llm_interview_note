# llm推理优化技术

> 原文链接：[Mastering LLM Techniques: Inference Optimization | NVIDIA Technical Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/ "Mastering LLM Techniques: Inference Optimization | NVIDIA Technical Blog")

堆叠Transformer层以创建大型模型可以获得更好的准确性、few-shot学习能力，甚至在各种语言任务中具有接近人类的涌现能力。这些基础模型的训练成本很高，而且在推理过程中可能需要大量的内存和计算（经常性成本）。当今最流行的大型语言模型（LLM）的大小可以达到数百亿到数千亿个参数，并且根据用例的不同，可能需要摄入长输入（或上下文），这也会增加开销。

这篇文章讨论了LLM推理中最紧迫的挑战，以及一些实用的解决方案。读者应该对transformer架构和注意力机制有一个基本的了解。

# 1.理解LLM推理

## 1.1 预填充阶段或处理输入

## 1.2 解码阶段或生成输出

## 1.3 批处理（Batching）

## 1.4 KV缓存

## 1.5 LLM内存需求

# 2.模型并行化扩展LLM

## 2.1 Pipeline并行

## 2.2 Tensor并行

## 2.3 Sequence并行

# 3.注意力机制优化

## 3.1 多头注意力（MHA）

## 3.2 多查询注意力（MQA）

## 3.3 分组注意力（GQA）

## 3.4 Flash attention

# 4.KV缓存的分页高效管理

# 5.模型优化技术

## 5.1 量化（Quantization）

## 5.2 稀疏（Sparsity）

## 5.3 蒸馏（Distillation）

# 6.模型服务技术

## 6.1 In-flight批量（**In-flight batching**）

## 6.2 推测推理（**Speculative inference**）

# 7.结论
