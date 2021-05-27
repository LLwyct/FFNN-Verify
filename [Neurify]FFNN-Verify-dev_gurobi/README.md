# FFNN-Verify
一个全连接ReLU节点的前馈神经网络的形式化验证器。使用了区间传播 + 混合整形线性规划两种方法进行神经网络验证。

- [FFNN-Verify](#ffnn-verify)
  - [1.1. dependence：](#11-dependence)
  - [1.2. example:](#12-example)
- [文件说明](#文件说明)
  - [v4.0](#v40)
  - [v4.1](#v41)
  - [v4.2](#v42)
  - [v4.2.1](#v421)
  - [v4.2.2](#v422)
  - [v4.3](#v43)
  - [v4.3.1](#v431)

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

## v4.1
这一次的更新，大量重构了之前的代码，减少了大量的冗余代码，添加了大量注释，减少了垃圾代码，提高了模块间的可复用性。并且实现了任意输出约束的自动添加，不必再为不同待验证属性手动添加输出约束。

- 添加了`ConstraintFormula`类，分为析取约束和合取约束
- 在`Layer.py`中删除了大量之前为了方便，添加的冗余代码
- 由于输出约束自动添加，在`Layer.py`中删除了大量冗余代码
- 输入约束在`property.py`写入，并由`initBounds` 函数添加，因此删除了大量冗余代码
- 为之前每个文件提供了README文件，并展示每一代的优化说明

在最新的此版本中，对于不同优化方案在`netWorkClass.py`中进行整合：

```python
# 初始化/预处理隐藏层及输出层的边界
# 0 MILP with bigM
# 1 MILP with ia  区间传播
# 2 MILP with sia 符号区间传播
# 4 MILP with slr 符号线性松弛
if GlobalSetting.preSolveMethod == 0:
    pass
elif GlobalSetting.preSolveMethod == 1:
    self.intervalPropation()
elif GlobalSetting.preSolveMethod == 2:
    self.symbolIntervalPropation_0sia_or_1slr(0)
elif GlobalSetting.preSolveMethod == 4:
    self.symbolIntervalPropation_0sia_or_1slr(1)
```

下一步的更新计划：

- bigM的M的大小在哪里预给出还没有说明
- 优化确定边界束紧程度的判断逻辑，该问题在v4.0的更新中提出，尚未解决
- 同上一点，一样需要加一些输出来表现该优化的提升在哪里的具体的数据化展示，之前虽然有做，但是我认为并不能直接在论文中使用，不够细节

## v4.2
- 更新了新的数据评判标准
- 更新了新的优化思路，结合slr和sia

## v4.2.1
优化了结合了slr与sia的代码，但是效果与想象的不是十分一致

理论来讲缩小了M的值，应该无论如何速度都更快，但是在1_1中速度很慢，在其他的里面速度有明显提升。

且发现，如果仅对某些层做优化，速度还能再提升的更快，例如p3 下的1-6，若仅对id=2的层做出入度区间优化，能跑到7s以内，slr需要34秒

## v4.2.2
在运行求解器时不打印多余信息

## v4.3
增加了Planet中提到的，通过线性近似和gurobi求解器去优化节点的上下界

## v4.3.1
修改了 indexToVar 没有更新的bug

现在的问题是，其实在gurobi模型中并没有节点入度上下界的信息，入度上下界只用于判断是添加哪种约束，如果上界小于0，添加node=0的约束，此类。真正被编码进模型中的参数只有出度的上下界，但此上下界的紧密与求解效率不呈现正相关。在fixed节点数不变的情况下缩紧入度边界是无效的，但是求解器却有严重的时间不一样的问题。

事实证明，slr+sia+opt > slr + opt > slr,这是运用了这些不同技术的入度的区间紧束程度。并且在p3 1_6 下，siaslropt甚至能多固定第二relu层的一个节点，但是，求解时间居然变长了，无法理解。

因此，这些小细节的优化无法明确提高求解效率，这与求解器的内部实现以及内部随机性相关，因此还是要在大优化上下功夫。