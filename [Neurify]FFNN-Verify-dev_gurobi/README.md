# FFNN-Verify
一个全连接ReLU节点的前馈神经网络的形式化验证器。使用了区间传播 + 混合整形线性规划两种方法进行神经网络验证。

- [FFNN-Verify](#ffnn-verify)
  - [1.1. dependence：](#11-dependence)
  - [1.2. example:](#12-example)

---

## 1.1. dependence：
- numpy
- gurobi
- keras
- onnx


## 1.2. example:
本项目以ACAS XU为例，检查了属性3在网络1_1到1_9下的验证结果。

```
python main.py --npath /Acas/acas_1_1.h5 --type rea --prop 3
```

如果要验证其他属性，请自行编写输出约束的反例。

说明：

|arg | description | example|
|:---:|:---:|:---:|
|--npath |神经网络文件的路径，以resources为根目录 | `/Acas/acas_1_1.h5`|
|--type | 验证属性的类型，可达性或局部鲁棒性，暂时只支持可达性验证 | `rea`|
|--prop | 针对ACAS XU的属性验证，从1到10 | `3` |

验证结果：

|network | result |
|:---:|:---:|
| acas_1_1.h5 | `sat` |
| acas_1_2.h5 | `sat` |
| acas_1_3.h5 | `sat` |
| acas_1_4.h5 | `sat` |
| acas_1_5.h5 | `sat` |
| acas_1_6.h5 | `sat` |
| acas_1_7.h5 | `unsat` |
| acas_1_8.h5 | `unsat` |
| acas_1_9.h5 | `unsat` |
# 文件说明

## v4.0
该版本为从前向后的不断发展的过程，该版本在上一个基于Reluval中符号传播的基础上（并没有实现区间分割），实现且仅实现了Neurify中的符号线性松弛，效果更好了。

但是奇怪的是，与Reluval相比，区间松弛程度增加了，但是可固定的relu节点数增加了，先考虑为先前判断区间束紧程度的思路不严谨，考虑重写该部分代码。

之前直接在 `out_bounds['ub']` 中取最大值作为区间的最大上界maxb，同理在 `out_bounds['lb']` 中取最小值作为区间的最小下界minb。该思路本意为判定bigM的具体值而考虑的，其中`M=max(|maxb|,|minb|)`。

看来现如今继续使用该思路与来判断边界的束紧程度不合适，因为有可能存在个别节点提供了非常大的上界或下界，但其实更多节点的束紧程度是增加了的。因此出现了上述奇怪的行为。
