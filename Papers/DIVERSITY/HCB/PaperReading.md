# Show Me the Whole World: Towards Entire Item Space Exploration for Interactive Personalized Recommendations
[Paper Link](https://arxiv.org/pdf/2110.09905.pdf)
## INTRODUCTION
推荐系统帮助用户从大量的候选中找到感兴趣的目标，其中非常典型的协同过滤以及 DeepFM 等通过用户的历史交互信息预测用户未来的行为，从而进行推荐。这种模式的推荐系统往往陷入一个闭环循环：用户大多只能与推荐的 item 产生交互；推荐系统考虑用户的历史行为进一步推荐相似 item。长此以往，整个系统会对每个用户形成信息茧。 <br>
类似 LinUCB 的多臂老虎机是一类经典的算法，这种方法利用辅助信息来平衡 exploration 和 exploitation，从而缓解闭环循环问题。在每个回合，从 K 个候选中选择最大潜在收益的一个方案，然后根据 user 与 item 的交互计算收益。最终的目标是经过 T 个回合后最大化累计收益。然而这些算法都存在共性的前提：K 是比较小时，那么通过枚举选取最优方案是可行的。这种假设只适用于候选有限情况。为了真正的探索用户的潜在兴趣减小信息茧房效应，候选方案通常由整个 item 空间构成，包含百万甚至亿万的 item，此时传统的多臂老虎机方案将会变的不再可行。 <br>
为了解决这个问题，作者提出了 hierarchical contextual bandit(HCB) 算法在大规模推荐场景下进行用户兴趣探索。树结构被广泛的应用到分割搜素空间从而达到减小搜索成本的场景下。HCB 采用树结构作为索引进行由粗到细的搜索，例如在电商场景下，“Apparel > Clothing > Women’s Clothing > Dresses” 是一个由 Apparel 到 Dresses 的路径。作者实现了一个基于 item embedding 的自下而上的聚类方法，并组成了一颗树，树种每个节点都是由语义相似的 item 构成。因此，与树中每个节点相关的 item 数量能够得以平衡，用户的相关行为也能够被编码到树中。HCB 利用分层信息将兴趣探索问题转化为决策问题。从树的根部开始，在非叶子节点上，HCB 在该节点的子节点上执行 bandit 算法去选择一个子节点，直至找到叶子结点。然后，再次使用 bandit 算法从叶子节点中选择 item 推荐给 user 并获取 reward，这个 reward 将会作为信号沿着搜索路径反向传导修正树结构对 user 兴趣的建模。 <br>
HCB 的流程类似于 DFS 算法。然而，以 DFS 方法选择路径时会引起不确定性，特别是对较深的树。首先，如果父节点的选择是不准确的，此后所有的选择都将会收到影响，作者将之称为 error propagation。其次，由于用户的兴趣往往是多样的，那么很有可能用户对树中不同位置的多个节点感兴趣。因此，作者进一步提出 progressive HCB(pHCB) 算法来减少不确定性改善推荐效果。类似 BFS 算法，pHCB 以一种自适应的自上而下的方法探索 item。一方面，这种方式维持有限数量的节点作为感受野。当一个节点被多次访问并被证明这个兴趣成立后，那么这个节点的子节点将会被包含到感受野中，并且这个节点将被移除。另一方面，pHCB 通过 bandit 算法在感受野中的节点上探索用户不同方面的兴趣。最后，相比于 HCB，pHCB 避免重复选取同一个节点中的 item。

## RELATED WORK
