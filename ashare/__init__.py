# -*- coding: utf-8 -*-
"""A股截面因子挖掘（Phase 1，定位 B：因子合成器）。

红线约束：唯一外部代码依赖为 GMtest 的 utils 数据层（经 gm_bridge 转引）；
宇宙过滤 / 标签 / 评估等一切逻辑在本仓库本地实现（契约复刻自
GMtest models/topics/cross_section 课题的文档约定，不 import 其代码）。
"""
