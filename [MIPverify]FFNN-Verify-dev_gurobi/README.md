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

说明：

|arg | description | example|
|:---:|:---:|:---:|
|--npath |神经网络文件的路径，以resources为根目录 | `/Acas/acas_1_1.h5`|
|--type | 验证属性的类型，可达性或局部鲁棒性，暂时只支持可达性验证 | `acas` or `mnist`|
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

本项目相比于`[NSVerify]`版本，使用且仅使用了区间算术(朴素区间传播)
