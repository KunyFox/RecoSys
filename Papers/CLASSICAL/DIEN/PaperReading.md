# Deep Interest Evolution Network for Click-Through Rate Prediction
[Paper Link](https://arxiv.org/pdf/2005.12981.pdf)

## Deep Interest Evolution Network
### Review of BaseModel
作者将从特征、模型结构以及损失函数等方面介绍 base model。<br>
__Feature Representation__ 线上的推荐系统使用了四种特征：_User Profile, User Behavior, Ad_ and _Context_，值得注意的是 ad 也就是 item。每个种类的特征都有几个分组，User Profile 可以分为 gender、age 等；ad 特征可以分为 ad_id、shop_id 等；Context 可分为 devise type id, time 等，每个分组的特征都被编码为 one-hot 向量，User Profile, User Behavior, Ad 和 Context 中每个分组的 one-hot 向量拼接得到 $x_p, x_a, x_c. x_b$。在 CTR 预估模型中，每个分组包含一系列的行为，每个行为队形一个 one-hot 向量，如下：<br>
<div align="center"><img src=pics/eq1.png></div>

其中 $b_t$ 为第 $t$ 个行为的 one-hot 向量，$T$ 是用户的的历史行为数量，$K$ 是用户能够产生交互的商品数量。<br>
__The Structure of BaseModel__ 大多数 CTR 预估模型都采用 embedding & MLP，主要包含一下几个部分：<br>

- __Embedding__ ：Embedding 是一种将高维稀疏特征转换为低维稠密特征的通用手段，在 embedding 层中，每个特征组都对应着一个特征矩阵。例如，用户浏览过的商品可以表示为 $E_{good} = [m_1; m_2; ...; m_K] \in R^{n_E \times K}$，其中 $m_j \in R^{n_E}$ 表示维度为 $n_E$ 的特征向量。 若 $b_t[j_t] = 1$，那么其对应的特征向量为 $m_{j_t}$，用户的行为特征向量序列可以表示为 $e_b = [m_{j_1};m_{j_2};...;m_{j_K}]$。
- __Multilayer Perceptron (MLP)__ 首先，同一类特征的特征向量会通过 pooling 压缩到一个向量中。然后所有通过 pooling 压缩的向量会通过拼接进行组合。最终，拼接后的特征会经过 MLP 进行最后的预测。

__Loss Function__ 在 CTR 预估模型中广泛使用的损失函数是对数似然函数，通过 item 的 label 对模型进行监督训练：<br>
<div align="center"><img src=pics/eq2.png></div>

其中 $x = [x_p, x_a, x_c, x_b] \in D$，$D$ 是容量为 $N$ 的训练集，$y \in \{0, 1\}$ 表示用户是否点击了 item，$p(x)$ 是网络的输出，代表用户点击 item 的概率。

### Deep Interest Evolution Network
不同于搜索，在许多展示广告的电商平台，用户不会直接展现自身的潜在兴趣，因此挖掘用户的潜在兴趣以及兴趣变化对 CTR 预估来说只管重要。DIEN 就是用来挖掘用户兴趣以及兴趣演变过程的网络。如 Fig1 所示，DIEN 由几部分构成：首先所有的特征都通过 embedding layer 进行编码。然后，DIEN 采用两个阶段去挖掘用户兴趣演变：interest extractor layer 从用户行为序列中抽取用户的兴趣；interest evolving layer 根据 target item 建模用户兴趣的演变。最后兴趣表征和 ad, user profile, context 的表征将会被拼接并使用 MLP 进行最终预测。<br>
<div align="center"><img src=pics/Fig1.png> </div>

__Interest Extractor Layer__ 在电商系统中，用户行为是用户兴趣的载体，用户行为发生变化后兴趣也会相应产生变化。用户的行为是及其丰富的，即便在很短的一段周期内，用户行为序列也能积攒的很长。为了平和效果和效率，作者使用 GRU 来建模行为之间的依赖性，GRU 的输入是根据时间排序的用户行为，GRU 涉及的公式如下：<br>
<div align="center"><img src=pics/eq3.png> </div>

其中 $\sigma$ 是 sigmoid 激活函数，$\circ$ 是元素相乘，$W^{u}, W^{r}, W^{h} \in R^{n_H \times n_I}$，$U^{z}, U^{r}, U^{h} \in R^{n_H \times n_H}$，$n_H$ 是隐藏层维度，$n_I$ 是输入维度。$i_t$ 是 GRU 的输入，$i_t = e_b[t]$ 代表用户的第 t 个行为，$h_t$ 是第 t 个隐藏状态。<br>
然而，隐藏状态 $h_t$ 只能捕捉行为之间的依赖性，但是不然计算兴趣的效率。由于用户最新的兴趣影响 target item 的点击，target item 的标签 $L_{target}$ 仅仅包含对最新兴趣的监督，而历史状态 $h_t(t < T)$ 没有恰当的监督训练。众所周知，兴趣的演变与行为的变化息息相关，因此作者提出 auxiliary loss，使用行为 $b_{t+1}$ 来监督兴趣状态 $h_t$ 的学习。除了使用下一个行为作为正样本，同时也从样本空间采样非点击样本作为负样本。<br>
有 $N$ 对行为序列：$\{e_b^{i}, \hat{e}_b^{i}\} \in D_B, i \in 1, 2, ..., N$，其中 $e_b^{i} \in R^{T \times n_E}$ 代表点击的样本序列，$n_E$ 是 embedding 维度，$e_b^{i}[t] \in G$ 表示用户点击的第 t 个 item 的 embedding，$\hat{e}_b^{i}[t]\ \in G - e_b^{i}[t]$ 在 t 时刻用户未点击的 item 的采样样本。Auxiliary loss 如公式 (7) 所示：<br>
<div align="center"><img src=pics/eq7.png> </div>

其中 $\sigma (x_1, x_2) = \frac{1}{1 + exp(-[x_1, x_2])}$ 是 sigmoid 激活函数，$h_t$ 代表 t 时刻的 GRU 隐藏状态，最终的 loss 如公式 (8)：
<div align=center><img src=pics/eq8.png></div>

在 auxiliary loss 的作用下，每个状态 $h_t$ 能够充分的表征 t 时刻用户进行 $i_t$ 行为时的兴趣，$[h_1, h_2, ..., h_T]$ 组成了兴趣序列，可以作为兴趣演进的输入。<br>
__Interest Evolving Layer__ 受到外部环境和内部认知的共同影响，用户的不同兴趣会随时间发生变化。建模兴趣变化的好处有以下几点：<br>

- 能够提供与用户最新兴趣相关的历史信息。
- 依照用户兴趣演变来预测用户点击更有依据。

兴趣在演变时体现了两种特质：

- 由于兴趣存在多样性导致兴趣会发生转移。兴趣转移现象在用户行为上表现为用户在一段时期可能对多种书籍感兴趣，而在另一段时间对衣服感兴趣。
- 兴趣之间可能相互影响，每个兴趣都有自身的演变过程。

在第一阶段，通过 auxiliary loss 获取到了用户的兴趣序列，通过对兴趣序列的分析，作者将 attention 结构的局部激活能力与 GRU 的序列学习能力结合到一起，以此来建模兴趣演化。在 GRU 每个 step 中的局部激活可以强化相关兴趣的作用，弱化兴趣转移的噪声，有助于针对 target item 建模兴趣转移。<br>
使用 $i^{'}_t, h^{'}_t$ 表示 interest evolving module 中 GRU 的输入和隐藏状态，第二个 GRU 的输入与 Interest Extractor Layer 的兴趣状态相关：$i^{'}_t = h_t$。最后的隐藏状态 $h_T$ 代表着最新的兴趣。<br>
在 interest evolving module 中使用的 attention 如公式 (9) 所示：
<div align="center"><img src=pics/eq9.png> </div>

接下来介绍几种 attention 与 GRU 的组合方式：

- __GRU with attentional input (AIGRU)__ 为了能够在兴趣演化中激活相关兴趣，作者提出 AIGRU 的方法，用 attention score 影响 interest evolving layer 的输入，如公式 (10)。其中 $h_t$ 是 interest extractor layer 中 GRU 的第 t 个隐藏状态，$i^{'}_t$ 用做兴趣演化的第二个 GRU 输入，* 表示向量积。在 AIGRU 中，attention score 可以抑制相关度较小的兴趣。然而，AIGRU 的效果并不是特别好，因为零输入也能改变 GRU 的状态，相关度较小的兴趣依然能够影响兴趣演化的建模。
<div align="center"><img src=pics/eq10.png> </div>

- __Attention based GRU(AGRU)__ 通过 attention 机制对 GRU 输入信息进行编码，AGRU 能够从复杂的序列中有效地提取关键信息。AGRU 使用 attention score 代替 GRU 的更新门，并且改变隐藏状态的计算机制，如公式 (11)。其中 $h^{'}_t, h^{'}_{t-1}$ 为 AGRU 的隐藏状态，AGRU 使用 attention score 控制隐藏状态的更新，弱化了相关性较小的兴趣的影响。
<div align="center"><img src=pics/eq11.png></div>

- __GRU with attentional update gate (AUGRU)__ AGRU 使用标量 $a_t$ 代替原本的更新门向量 $u_t$ 忽略了不同维度的重要性不同的影响，作者提出 AUGRU 将 attention 机制与 GRU 相结合，如公式 (12) 和 (13) 所示。其中 $u^{'}_t$ 是 AUGRU 原始的更新门，$\tilde{u}^{'}_t$ 是使用 attention 机制更新后的更新门，$h^{'}_t, h^{'}_{t-1}, \tilde{h}^{'}_t$ 是隐藏状态。AUGRU 保持了原始的更新门的维度，可以决策每个维度的重要性。由于信息的差异化，AUGRU 使用 attetion score $a_t$ 加权更新门，使得相关度较小的兴趣的影响显著下降。
