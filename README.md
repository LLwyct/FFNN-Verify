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


## 1.2. example:
本项目以ACAS XU为例，检查了属性3在网络1_1到1_9下的验证结果。

```
python main.py --npath /Acas/acas_1_1.h5 --type rea --prop 3
```

如果要验证其他属性，请自行编写输出约束的反例。