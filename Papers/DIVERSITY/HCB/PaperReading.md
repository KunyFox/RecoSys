# Show Me the Whole World: Towards Entire Item Space Exploration for Interactive Personalized Recommendations
[Paper Link](https://arxiv.org/pdf/2110.09905.pdf)
## INTRODUCTION
推荐系统帮助用户从大量的候选中找到感兴趣的目标，其中非常典型的协同过滤以及 DeepFM 等通过用户的历史交互信息预测用户未来的行为，从而进行推荐。这种模式的推荐系统往往陷入一个闭环循环：用户大多只能与推荐的 item 产生交互；推荐系统考虑用户的历史行为进一步推荐相似 item。长此以往，整个系统会对每个用户形成信息茧。 <br>
类似 LinUCB 的多臂老虎机是一类经典的算法，这种方法利用辅助信息来平衡 exploration 和 exploitation，从而缓解闭环循环问题。在每次迭代，从 K 个候选中选择最大潜在收益的一个方案，然后根据 user 与 item 的交互计算收益。最终的目标是经过 T 轮迭代后最大化累计收益。然而这些算法都存在共性的前提：K 是比较小时，那么通过枚举选取最优方案是可行的。这种假设只适用于候选有限情况。为了真正的探索用户的潜在兴趣减小信息茧房效应，候选方案通常由整个 item 空间构成，包含百万甚至亿万的 item，此时传统的多臂老虎机方案将会变的不再可行。 <br>
为了解决这个问题，作者提出了 hierarchical contextual bandit(HCB) 算法在大规模推荐场景下进行用户兴趣探索。树结构被广泛的应用到分割搜素空间从而达到减小搜索成本的场景下。HCB 采用树结构作为索引进行由粗到细的搜索，例如在电商场景下，“Apparel > Clothing > Women’s Clothing > Dresses” 是一个由 Apparel 到 Dresses 的路径。作者实现了一个基于 item embedding 的自下而上的聚类方法，并组成了一颗树，树种每个节点都是由语义相似的 item 构成。因此，与树中每个节点相关的 item 数量能够得以平衡，用户的相关行为也能够被编码到树中。HCB 利用分层信息将兴趣探索问题转化为决策问题。从树的根部开始，在非叶子节点上，HCB 在该节点的子节点上执行 bandit 算法去选择一个子节点，直至找到叶子结点。然后，再次使用 bandit 算法从叶子节点中选择 item 推荐给 user 并获取 reward，这个 reward 将会作为信号沿着搜索路径反向传导修正树结构对 user 兴趣的建模。 <br>
HCB 的流程类似于 DFS 算法。然而，以 DFS 方法选择路径时会引起不确定性，特别是对较深的树。首先，如果父节点的选择是不准确的，此后所有的选择都将会收到影响，作者将之称为 error propagation。其次，由于用户的兴趣往往是多样的，那么很有可能用户对树中不同位置的多个节点感兴趣。因此，作者进一步提出 progressive HCB(pHCB) 算法来减少不确定性改善推荐效果。类似 BFS 算法，pHCB 以一种自适应的自上而下的方法探索 item。一方面，这种方式维持有限数量的节点作为感受野。当一个节点被多次访问并被证明这个兴趣成立后，那么这个节点的子节点将会被包含到感受野中，并且这个节点将被移除。另一方面，pHCB 通过 bandit 算法在感受野中的节点上探索用户不同方面的兴趣。最后，相比于 HCB，pHCB 避免重复选取同一个节点中的 item。

## PRELIMINARY AND PROBLEM

<div align="center"> <img src=pics/Table1.png> </div>

### 1.UCB for Recommender Systems
推荐系统可以视为是一种决策，包含 M 个 user 和 N 个 item。在每个交互迭代 $t = 1, 2, ..., T$，对用户 $u$，通过策略 $\pi$ 推荐 item $i_{\pi}(t)$，并收到相应的反馈 $r_{\pi}(t)$。例如，若用户 $u$ 对 $i_{\pi}(t)$ 进行点击，那么 $r_{\pi}(t)$ 为 1，否则为 0。最优的策略为 $\pi^{*}$，目标是学习最优策略 $\pi$，经过 T 轮迭代积累的损失如下：

<div align="center"> <img src=pics/eq1.png> </div>

实际上，由于最优策略 $\pi^*$ 是未知的，通过最大化 $\sum_{t=1}^{T}E(r_{\pi}(t))$ 来等价代替最小化 Eq1。 <br>
bandit 算法的核心是寻找 exploration 和 exploitation 的平衡，这样系统将会有确定的机会去探索用户兴趣，同时也不会浪费过多的资源在用户不感兴趣的候选上。参考 LinUCB，每个 item 没视为是一个 arm，在第 t 轮迭代接收到 user 的请求后，通过 Eq2 选取 arm 策略 $a_{\pi}(t)$：
<div align="center"> <img src=pics/eq2.png> </div>

LinUCB 的策略 $\pi$ 是关于特征向量 $x_a$ 和用户参数 $\theta_u$ 的线性函数，reward 计算方式为 $R_a(t) = \theta_u^Tx_a + \eta$，$\eta$ 是一个均值为0方差小于1的高斯随机数，表示环境噪声。上界 $C_a(t)$ 衡量 reward 的不确定性。计算的关键之处在于如何确定 $\theta_u$ 以及上界 $C_a(t)$。根据 LinUCB 可知：
<div align="center"> <img src=pics/eq3.png> </div>

其中 $D_t \in R^{t \times d}$ 是第 t 次迭代的交互臂的特征矩阵，$\alpha$ 是控制上界的超参数，$r_t \in R^t$ 是第 t 次迭代的用户响应向量。

### 2.The Challenges
LinUBC 需要对每个 arm 均进行计算 reward 然后选取其中最优的。在现代推荐系统中，item 的数量是巨大的，因此在科研领域会从总量为 N 的 arm 中随机选取 K 个，组成小的 arm 集合 $A_t$ 并执行 LinUCB 算法；在工业领域，bandit 算法只能使用在候选数目较小的场景下。作者认为，将 bandit 算法放在 item 召回阶段是能够更好的挖掘用户的潜在兴趣的，因为在召回阶段候选是完整的候选集。而在后续的排序阶段，候选集是经过筛选的并且与用户的历史行为具有高度相关性的。因此，在推荐系统的排序阶段挖掘用户潜在兴趣的意义是相对较小的。为了尽量避免信息茧效应，作者提倡在完全的候选集上挖掘用户潜在兴趣。然而目前没有将 bandit 算法应用在大规模候选集上的工作。 <br>
为了解决这个问题，作者提出用树结构去整个候选空间，并构建候选之间的分层依赖来加速兴趣探索。定义如下： <br>

__Framework 1. *Tree-based Exploration*__ 整个候选空间可以被结构化为一颗树 $H$，树中的节点都是具有相同特质的 item 集合，从根部到叶子结点的过程是这些特质逐渐细分的过程。在探索潜在兴趣的过程中，首先会通过一些机制选取一个节点，然后从这个节点对应的候选集中选取一个 item，然后根据用户的反馈更新 item-wise 和 node-wise 的路径参数。

## METHODOLOGY
### 1.Tree Structure Construction
在设计 hierarchical bandit algorithms 过程中，树的结构起到了重要作用。item 的类别信息可以被用来构造树结构，然而这种方式存在两处不足，一方面由于不同类的之间的数量差距较大，存在严重不均衡现象；另一方面这种构造方式没有利用到 user 行为信息，简单的使用类别信息会造成性能下降。基于此，作者首先通过 item 信息和 user 的点击相关信息训练 item embedding，然后使用 K-Means 聚类算法自下而上地构建一颗分层树结构，以此来建模 item 之间的相关性。 <br>
