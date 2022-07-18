# Deep Interest Network for Click-Through Rate Prediction (DIN)
[Paper Link](https://arxiv.org/pdf/1706.06978.pdf)

## BACKGROUND
在 Alibaba 等电商网站，广告是一种天然的货品。这篇 paper 中，在没有额外的说明时，我们将广告视为一种货品。
<div align=center><img src=pics/Figure1.png></div>
如 Fig1 所示，Alibaba 的广告系统有两阶段构成：<br>
1）匹配阶段：通过协同过滤等方法根据用户信息生成候选集；<br>
2）排序阶段：对候选广告预测 CTR 并选出头部候选。<br>
每天有大量用户访问电商网站并留存下大量的用户行为数据，这对匹配和排序具有重要意义。具有丰富行为历史的用户同时具有多样的兴趣。例如，一个年轻的妈妈最近浏览了 woolen coat, T-shits, earrings, tote bag, leather handbag 以及 children’s coat，这些行为数据能够表达出她的购物兴趣，并依此在她访问电商网站时推荐合适的广告，例如 handbag。显然，广告的展示只能匹配这位年轻妈妈的一部分兴趣。总而言之，具有丰富历史行为的用户兴趣是多样性的(diverse)并且应该被部分激活(locally activated)从而获得准确的广告推荐。

## DEEP INTEREST NETWORK
不同于搜索，用户在点击广告时并没有表现出明确的意图，因此在构建 CTR 模型时需要有效的方法从用户大量的行为数据中寻找出用户的兴趣.广告系统中最基础的是用来描述用户自身和广告本身的特征，关键之处在于合理的利用这些特征并且挖掘有用信息。

### 1.Feature Representation
在 CTR 预测任务中，数据通常以分组的形式存在，例如：[weekday=Friday, gender=Female, visited_cate_ids={Bag,Book}, ad_cate_id=Book]，这些数据通常被编码成高维的稀疏向量形式。第 i 个特征分组的编码向量表示为 $t_i \in R^{K_i}$ ，$K_i$ 表示第 i 个特征组中不同 id 的数量也是特征的维度。$t_i[j]$ 表示 $t_i$ 中的第 j 个元素，并且 $t_i[j] \in \{0, 1\}$。并有 $\sum_{j=1}^{K_i}t_i[j] = k$，当 k = 1 时为 one-hot 编码，当 k > 1 是为 multi-hot 编码。那么所有特征组的总特征可以表示为 $x = [t^T_1, t^T_2, ...t^T_M]^T$，其中 M 为特征组的数量，$\sum_{i=1}^{M}K_i = K$，K 为整个特征空间的维度。 <br>
<div align=center> <img src=pics/Table1.png> </div>
如 Table1 所示，在系统中所使用的特征集合由 4 部分构成，其中 user behavior features 是典型的 multi-hot 编码并且包含了丰富的用户兴趣信息。值得注意的是作者没用使用组合特征，并通过 DNN 来抽取交互特征。

### 2. Base Model(Embedding&MLP)
大多数的模型结构遵循一种 Embedding&MLP 的范式，如 Fig2 中左边所示，作者将其作为实验的 base model。这种范式由以下几部分组成： <br>
**Embedding layer**：由于输入的特征都是高维的稀疏向量，embedding layer 将其转换为低维的稠密向量。以第 i 个特征组的特征向量 $t_i$ 为例，以 $W^i = [w^i_1, w^i_2, ..., w^i_{K_i}] \in R^{D \times K_i}$ 表示第 i 个特征组的特征字典，其中 $w^i_j \in R^D$ 表示维度为 D 的特征向量。从而 embedding 操作可以视为一种查表操作。 <br>

- 如果 $t_i$ 是 one-hot 编码并且 $t_i[j] = 1$，那么 $t_i$ 的特征向量是单一表示：$e_i = w^i_j$。
- 如果 $t_i$ 是 multi-hot 编码并且 $t_i[j] = 1, j \in {i_1, i_2, ..., i_k}$，那么 $t_i$ 的特征向量是一组向量 list 表征：$\{e_{i_1}, e_{i_2}, ..., e_{i_k}\} = \{w^i_{i_1}, w^i_{i_2}, ..., w^i_{i_k}\}$。
  
**Pooling layer and Concat layer**：由于用户之间的历史行为数量不同，导致 multi-hot 编码的行为特征 $t_i$ 的维度无法确定。为了能够使用全连接网略进行特征提取，通常采用 pooling 层来获得固定维度的输入：
<div align=center> <img src=pics/eq1.png> </div>
其中最常用的两种方式就是 sum pooling 和 average pooling，这两种 pooling 方式会对特征向量 list 做元素级的加法/乘法。 <br>
Embedding layer 和 pooling layer 都是对所有特征组操作，将原本系数的高维特征映射为固定维度的低维的稠密特征，然后将所有的特征做拼接得到最终的表征。 <br>
**MLP**：在拿到拼接后特征后使用全连接层学习组合特征。
**Loss**：base model 所使用的损失函数为交叉熵
<div align=center> <img src=pics/eq2.png> </div>
其中 S 是容量为 N 的训练集，x 为模型输入，y 为样本标签，p(x) 为模型输出。

<div align=center> <img src=pics/Figure2.png> </div>

### 3. The structure of Deep Interest Network
在建模用户兴趣过程中，user behavior 特征起到非常关键的作用。Base model 通过 pooling 操作获取用户行为的表征，这种方式得到的表征对于同一个用户来说是没用变化的，忽略了候选广告的影响，并且无法表现出用户兴趣的多样性。为了解决这个问题，有一个简答的方式是扩展用户行为表征的维度，但是这种方式会导致模型参数的激增，并且在训练过程中导致过拟合现象。 <br>
相比于使用固定表征表示用户兴趣，Deep Interest Network(DIN) 能够通过目标广告和用户行为自适应计算兴趣表征，这个表征会随着目标广告发生变化。 <br>
Fig2 中右边展示了 DIN 的结构，DIN 在 base mode 的基础上引入了 local activation unit 结构，通过目标广告 A 计算出 user behavior list 中每个特征的权重并进行 weighted sum pooling 得到用户表征 $v_U$，如 eq3 所示。
<div align=center> <img src=pics/eq3.png> </div>

其中 $\{e_1, e_2, ..., e_H\}$ 是用户 $U$ 的长度为 $H$ 行为表征序列，$v_A$ 是目标广告 A 的表征。这种方式得到的 $v_U(A)$ 会随着目标广告发生变化。$a(\cdot)$ 是一个网络，其输出最为 activation weight。如 Fig2 所示，除了两个输入表征外，$a(\cdot)$ 增加了二者外积信息作为后续网络的输入，从而进行显示的相关性建模。 <br>
Local activation 采用了和 NMT 中提出的 attention 相似的方法，然而和传统 attention 不同的是，为了能够保留用户兴趣的强度，$\sum_{i}w_i = 1$ 的约束被放开，并且$\sum_{i}w_i$ 的值也能够在一定程度上表示了用户被激活的兴趣强度。

## TRAINING TECHNIQUES
### 1.Mini-batch Aware Regularization
作者提出 Mini-batch Aware Regularization，这种方法仅计算 mini-batch 中使用到的稀疏特征的 L2-norm。事实上，CTR 模型中的 embedding 字典贡献了绝大部分的参数和计算量。以 $W \in R^{D \times K}$ 表示 embedding 字典的参数，其中 D 为 embedding vector 的维度，K 为特征空间的维度。那么对 $W$ 的 l2 regularization 可以表示为：
<div align=center> <img src=pics/eq4.png> </div>

其中 $w_j \in R^D$ 表示第 j 个 embedding vector，$I(x_j \ne 0)$ 表示 x 是否包含特征 id $j$，$n_j$ 表示包含特征 id $j$ 的总数量，Eq4 可以转换为 Eq5 得到 mini-batch aware regularization：
<div align=center> <img src=pics/eq5.png> </div>

其中 B 表示 mini-batch 的数量，$B_m$ 表示第 m 个 mini-batch。以 $a_{mj} = max_{(x,y) \in B_m}I(x_j \ne 0)$ 表示在 $B_m$ 中至少有一个 sample 有特征 id $j$，那么 Eq5 可以进一步表示为：
<div align=center> <img src=pics/eq6.png> </div>

### 2.Data Adaptive Activation Function
<div align=center> <img src=pics/Figure3.png> </div>

PReLU 在 0 点处有明显的跳变，如 Fig3 所示，这对每层输入特征分布不同的情况并不适用，因此作者设计了 Dice 激活函数。
<div align=center> <img src=pics/eq9.png> </div>

在训练阶段，E[s] 和 Var[s] 是 mini-batch 的输入特征的均值和方差，在测试阶段，E[s] 和 Var[s] 通过滑动平均计算。Dice 可以看成是 PReLU 的一般化，其思路是根据输入特征的分布情况进行调整，当 E[s]=0 和 Var[s]=0 时 Dice 退化为 PReLU。