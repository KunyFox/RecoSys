# Deep Interest Network for Click-Through Rate Prediction (DIN)
[Paper Link](https://arxiv.org/pdf/1706.06978.pdf)

## BACKGROUND
在 Alibaba 等电商网站，广告是一种天然的货品。这篇 paper 中，在没有额外的说明时，我们将广告视为一种货品。
<div align=center><img src=pics/Figure1.png></div>
如 Figure1 所示，Alibaba 的广告系统有两阶段构成：\
1）匹配阶段：通过协同过滤等方法根据用户信息生成候选集；\
2）排序阶段：对候选广告预测 CTR 并选出头部候选。\
每天有大量用户访问电商网站并留存下大量的用户行为数据，这对匹配和排序具有重要意义。具有丰富行为历史的用户同时具有多样的兴趣。例如，一个年轻的妈妈最近浏览了 woolen coat, T-shits, earrings, tote bag, leather handbag 以及 children’s coat，这些行为数据能够表达出她的购物兴趣，并依此在她访问电商网站时推荐合适的广告，例如 handbag。显然，广告的展示只能匹配这位年轻妈妈的一部分兴趣。总而言之，具有丰富历史行为的用户兴趣是多样性的(diverse)并且应该被部分激活(locally activated)从而获得准确的广告推荐。
## DEEP INTEREST NETWORK
不同于赞助搜索，