# Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks 笔记

emnlp2019
[https://arxiv.org/pdf/1908.10084.pdf](https://arxiv.org/pdf/1908.10084.pdf)
[https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)


# IDEA

1. BERT 原始的计算两个 sentence 的相似度问题时，是将 [CLS] sentence_1 [SEP] sentence_2 一起输入进 BERT，然后预测是否相似；论文提出这样的方式在线上推理时过于耗时，给的例子是有 1w 个句子，找其中最相似的句对，那么需要算 9999*10000/2 次，而 Sentence BERT 只需要 1w 次
1. 没有微调过的 BERT 直接拿它的输出做平均句向量，然后衡量相似性，效果比 glove 直接平均还要差；因此引入了 siamese 和 triplet network 的结构做微调
# Model
预训练模型可以为 BERT 或者 RoBERTa


单条句子过了 BERT 后，对输出做 pooling 操作，有三种方式：

1. 取 CLS token 点位输出的向量
1. 对所有的点位输出求平均
1. 对所有的点位输出求最大；max-over-time



使用 siamese 或者 triplet 结构来更新权重，即共享参数


目标函数有三种：

1. 分类目标函数
1. 回归目标函数
1. Triplet 目标函数



## 分类
u 和 v 分别是两个句子经过 BERT 并 pooling 后的向量，目标函数有：
![](https://cdn.nlark.com/yuque/__latex/b10ca0a36070e33bedeb6774811a0a9b.svg#card=math&code=o%3Dsoftmax%28W_%7Bt%7D%28u%2Cv%2C%7Cu-v%7C%29%29&height=20&width=223)


![](https://cdn.nlark.com/yuque/__latex/27524931e43f6cb8908851b123303c79.svg#card=math&code=%7Cu-v%7C&height=20&width=47) 是 element-wise 的操作，![](https://cdn.nlark.com/yuque/__latex/c158364343853921b55a4c76a01e6e33.svg#card=math&code=W_%7Bt%7D%5Cin%20R%5E%7B3n%5Ctimes%20k%7D&height=21&width=85)，其中 n 是句向量维度，k 是分类维度，然后做交叉熵训练
## 回归
u 和 v 做 cosine 相似度计算，损失函数为 mean squared error
## Triplet
三个句子，对于句子 a，它有一个 positive 的句子 p，一个 negative 的句子 n，loss 为：
![](https://cdn.nlark.com/yuque/__latex/a571974acde881fc0f83b8aacaec1965.svg#card=math&code=%5Cmax%28%7C%7Cs_%7Ba%7D-s_%7Bp%7D%7C%7C-%7C%7Cs_%7Ba%7D-s_%7Bn%7D%7C%7C%2B%5Cepsilon%20%2C0%29&height=21&width=251)
即和正例的距离要小于和负例的距离


