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
要构造一颗 $L$ 层的树，首先要将根据 item 的 embedding 相似性将其聚类到 $k_L$ 节点中，然后这些节点将会通过 K-Means 进一步被聚类到 $k_{L-1}$ 节点中，并与 $k_L$ 几点构成父子关系。重复以上步骤直至构建完成 $L$ 层的树。这样，这颗 $L$ 层的树 $H$ 将会在每一层分别包含 ${k_0, k_1, ..., k_L}$ 个节点，其中 $k_0 = 1$ 因为在第一层只有一个根节点，在同一个节点内的 item 彼此之间具有较大相似性。 <br>
### 2.Hierarchical Contextual Bandit
HCB 可以被泛化到不同的 bandit 算法中，此处作者以 LinUCB 为例进行展示。有两种 arm：$H$ 上的非叶子节点以及叶子结点。$H$ 中的每个节点代表确定的 item 分组，叶子结点的特种采用节点内部 item embedding 的 avg pooling 表示，非叶子结点的特征采用子节点的特征的 avg pooling 表示。HCB 算法是从根节点到叶子结点的决策过程。对于 $l$ 层的非叶子节点 $n^{(l)}(t)$，策略 $\pi$ 通过假设 reward 与节点的特征向量成线性关系：$\theta_u^{(l)T}X_n$，并依此从子节点 $Ch(n^{(l)}(t))$ 中选取最优。其中 $\theta_u^{(l)}$ 是用户 u 在第 $l$ 层的 latent 参数，$D^{(l)} \in R^{m \times d}$ 是第 $l$ 层 item 的特征矩阵，$D^{(l)}$ 的每一行都是 item 的特征向量，使用 ridge regression 进行系数预估： <br>
<div align="center"> <img src=pics/eq5.png> </div>

其中 $I$ 是单位矩阵，$r^l$ 是第 $l$ 层的历史 reward 向量。LinUCB 同时也考虑了置信区间以便能够更好预估收益。令 $A^{(l)} = D^{(l)T}D^{(l)} + I$，根据[32]，在置信度为 $1 - \delta$ 时上界可表示为：<br>
<div align="center"> <img src=pics/eq6.png> </div>

其中 $\delta > 0$ 并且 $\alpha = 1 + \sqrt{ln(2/\delta)/2}$。LinUCB 通过 Eq7 选取arm： <br>
<div align="center"> <img src=pics/eq7.png> </div>

若策略 $\pi$ 选择了 $i_{\pi}(t)$ 并获得了 reward 为 $r_{\pi}(t)$，那么由根节点到当前节点的整个路径 Path 上的节点都会获得 reward。因此，所有被选中的节点的 reward 都能够得到，那么可以更新参数 $\{\theta_u^{(0)}, \theta_u^{(1)}, ..., \theta_u^{(L)}\}$：<br>
<div align="center"> <img src=pics/eq8.png> </div>

其中 $A^{(l)}$ 和 $b^{(l)}$ 分别使用 d 维的单位矩阵和零向量初始化，$X^{(l)}$ 是第 $l$ 层别选择节点的特征。<br>

<div align="center"> <img src=pics/HCB.png> </div>

HCB 算法的伪代码如 Algorithm 1，作者提供了一个示例图 Fig1，一个三层的树结构，策略依次进行三次决策并最终选择了 $\{A, C, I, P\}$ 这条路径，然后使用另一个 bandit 算法从叶子结点 $p$ 中选择 item。被选择的 item 的 reward 会通过更新 reward 历史 $r^{(*)}$ 和交互历史 $D^{(*)}$ 影响树中参数 $\{\theta_u^{(0)}, \theta_u^{(1)}, \theta_u^{(2)}, \theta_u^{(3)}\}$ 的优化。 <br>
<div align="center"> <img src=pics/Fig1.png> </div>

### 3.Progressive Hierarchical Contextual Bandit
HCB 通过一系列的决策来学习用户兴趣并总是从叶子结点选取 item 会造成两个问题：(1)高层节点的决策很少能够影响到底层节点的决策，一旦在策略在某一层做了不好的选择，那么此后的选择都将是非最优的，当树层次结构更深时，问题尤其如此，作者将其命名为 _error propagation_；(2)用户可能不仅仅对一个节点感兴趣，贪心的选择一个节点将无法全面地捕捉用户兴趣。因此，作者进一步提出 pHCB 算法，核心的思想是根据历史曝光的反馈从上到下的扩大感受野。<br>
Definition 1. _Receptive field 是个性化的节点集合代表着当前用户可以被挖掘的潜在兴趣。在首次迭代，感受野只包含根节点，随着探索的进行，当满足了预先设定的条件后感受野将会被扩展。感受野中的节点被称为 visible nodes_。<br>
在 HCB 中只有叶子结点对应着一组 item，不同的是，在 pHCB 中允许策略在非叶子节点中选取相关联的 item。非叶子节点的 item 集合定义如下：<br>
Definition 2. _给定一个非叶子结点以及其子节点 $Ch(n)$，该节点对应的 item 集合定义为其叶子结点对应 item 集合的并集_。<br>
在第 $t$ 次迭代中，用户 $u$ 的感受野用 $V_u(t)$ 来表示，pHCB 将 $V_u(t)$ 中每个节点视作为一个 arm，通过 Eq2 计算最大的预估 reward 来选取一个 arm 记 $n(t)$，pHCB 直接从感受野中选取 arm 而不是沿着决策路径选取。<br>
<div align="center"> <img src=pics/Fig2.png> </div>

Fig2 是 pHCB 的一个示例，在 $T_a$ 次迭代，用户 $u$ 的感受野包含 $B,C,D$ 三个节点，在接下来的几次迭代，如果节点 $C$ 被多次选中并接收到正向反馈，那么它就满足了兴趣探索的条件，它的子节点 $G,H,I$ 将会代替 $C$ 被加入到感受野中。那么在第 $T_b$ 次迭代，感受将变为 $B,D,G,H,I$，用这种方式 pHCB 可以扩展自身感受野从而能够更好的探索用户兴趣。<br>
扩张感受野 pHCB 的关键机制，树中节点的聚合粒度不同，高层节点代表了更广泛的用户兴趣，而底层节点代表了更细致的用户兴趣。作者希望感受野能够快速的收敛到叶子节点，因此设置了如下的探索条件：对树 $H$ 的第 $l$ 层的非叶子节点 $n$, 若 (1)它被选择了至少 $qlogl$ 次而且(2)平均 reward 大于 $plogl(0 \le p \le 1)$，那么就使用其叶子结点代替它加入感受野，$p、q$ 均为超参数，$logl$ 使得高层的节点比底层节点更容易被探索，如 Algorithm 2。<br>
<div align="center"> <img src=pics/pHCB.png> </div>

