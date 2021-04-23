import numpy as np
from numpy import ndarray
from typing import Union, List, Dict, Optional
from gurobipy import Model
from options import GlobalSetting

class Layer:
    def __init__(self, layer_type="unknown"):
        # "linear" | "relu" | "input" | "unknown"
        self.type: str = layer_type
        self.size: Optional[int] = None
        self.var_bounds_in: Dict[str, Optional[ndarray]] = {
            "ub": None,
            "lb": None
        }
        self.var_bounds_out: Dict[str, Optional[List]] = {
            "ub": None,
            "lb": None
        }
        self.var = None
        self.nodeList = []

    def setVar(self, varlist):
        if self.var is None:
            self.var = np.array(varlist)
        else:
            pass

    def getVar(self):
        if self.var is not None:
            return self.var
        else:
            pass

    def setSize(self, size):
        self.size = size


class InputLayer(Layer):
    def __init__(self, layer_type="input", size=0):
        super(InputLayer, self).__init__(layer_type)
        self.type = layer_type
        self.size = size

    def setBounds(self, lb: ndarray, ub: ndarray):
        self.var_bounds_in["ub"] = self.var_bounds_out["ub"] = ub
        self.var_bounds_in["lb"] = self.var_bounds_out["lb"] = lb

class OutputLayer(Layer):
    def __init__(self, layer_type="output", size=0):
        super(OutputLayer, self).__init__(layer_type)
        self.type = layer_type
        self.size = size


class ReluLayer(Layer):
    def __init__(self, w: ndarray, b: ndarray, layer_type: str = "relu"):
        super(ReluLayer, self).__init__(layer_type)
        self.type:      str     = layer_type
        self.size:      int     = b.size
        self.weight:    ndarray = w
        self.bias:      ndarray = b
        self.reluVar:   ndarray = np.empty(self.size)

    def addConstr(self, preLayer: Layer, gmodel: Model):
        wx_add_b = np.dot(self.weight, preLayer.var) + self.bias
        if GlobalSetting.constrMethod == 0:
            '''
            0代表使用带激活变量的精确松弛
            1代表使用三角松弛
            '''
            # 如果self.var_bounds_in["lb"] 为 None说明没有使用区间传播，用的是大M
            # if self.var_bounds_in["lb"] is None:
            #     # 这里设置bigM的M的值
            #     M = 99
            #     print("solve with BigM=", M)
            #     for curNodeIdx, curNode in enumerate(self.var):
            #         # 1
            #         gmodel.addConstr(curNode >= wx_add_b[curNodeIdx])

            #         # 2
            #         gmodel.addConstr(curNode <= wx_add_b[curNodeIdx] + M * (1 - self.reluVar[curNodeIdx]))

            #         # 3
            #         gmodel.addConstr(curNode >= 0)

            #         # 4
            #         gmodel.addConstr(curNode <= M * self.reluVar[curNodeIdx])
            #     return
            # 这个变量用于计算区间传播所带来的边界束紧，对于每一层分别减少了多少binary变量
            ignoreBinaryVarNum = 0
            # 用这两个变量来测试并打印在区间传播的过程中，每一隐藏层Relu节点输入的上界的最大值和下界的最小值，从而判断自定义的M是否满足要求。
            maxUpper = -999
            minLower = 999
            print("solve with Interval Algorithm=")
            for curNodeIdx, curNode in enumerate(self.var):
                if self.var_bounds_in["ub"][curNodeIdx] > maxUpper:
                    maxUpper = self.var_bounds_in["ub"][curNodeIdx]
                if self.var_bounds_in["lb"][curNodeIdx] < minLower:
                    minLower = self.var_bounds_in["lb"][curNodeIdx]
                if self.var_bounds_in["lb"][curNodeIdx] >= 0:
                    gmodel.addConstr(curNode == wx_add_b[curNodeIdx])
                    ignoreBinaryVarNum += 1
                elif self.var_bounds_in["ub"][curNodeIdx] <= 0:
                    gmodel.addConstr(curNode == 0)
                    ignoreBinaryVarNum += 1
                else:
                    upper_bounds = self.var_bounds_in["ub"][curNodeIdx]
                    lower_bounds = self.var_bounds_in["lb"][curNodeIdx]
                    # 注意这里要根据不同的网络来调整
                    if upper_bounds > 99:
                        upper_bounds = 99
                    if lower_bounds < -999:
                        upper_bounds = -999
                    # 1
                    gmodel.addConstr(curNode >= wx_add_b[curNodeIdx])

                    # 2
                    gmodel.addConstr(curNode <= wx_add_b[curNodeIdx] - lower_bounds * (1 - self.reluVar[curNodeIdx]))

                    # 3
                    gmodel.addConstr(curNode >= 0)

                    # 4
                    gmodel.addConstr(curNode <= upper_bounds * self.reluVar[curNodeIdx])
            print(self.type, self.size, ignoreBinaryVarNum)
            print('maxupper', maxUpper)
            print('minLower', minLower)
        elif GlobalSetting.constrMethod == 1:
            for curNodeIdx, curNode in enumerate(self.var):
                if self.var_bounds_in["lb"][curNodeIdx] >= 0:
                    gmodel.addConstr(curNode == wx_add_b[curNodeIdx])
                elif self.var_bounds_in["ub"][curNodeIdx] <= 0:
                    gmodel.addConstr(curNode == 0)
                else:
                    # 1
                    gmodel.addConstr(curNode >= 0)

                    # 2
                    gmodel.addConstr(curNode >= wx_add_b[curNodeIdx])

                    # 3
                    tempk = self.var_bounds_in["ub"][curNodeIdx] / (self.var_bounds_in["ub"][curNodeIdx] - self.var_bounds_in["lb"][curNodeIdx])
                    gmodel.addConstr(
                        curNode <= (
                            tempk * (wx_add_b[curNodeIdx] - self.var_bounds_in["lb"][curNodeIdx])
                        )
                    )

    def setReluVar(self, reluVarList: List):
        self.reluVar = np.array(reluVarList)

    def getReluVar(self):
        if self.reluVar is not None:
            return self.reluVar
        else:
            pass

class LinearLayer(Layer):
    def __init__(self, w:ndarray, b:ndarray, layer_type: str = "linear"):
        super(LinearLayer, self).__init__(layer_type)
        self.type:      str     = layer_type
        self.size:      int     = b.size
        self.weight:    ndarray = w
        self.bias:      ndarray = b

    def addConstr(self, preLayer, gmodel: Model):
        if GlobalSetting.constrMethod == 0 or GlobalSetting.constrMethod == 1:
            wx_add_b = np.dot(self.weight, preLayer.var) + self.bias
            for curNodeIdx, curNode in enumerate(self.var):
                gmodel.addConstr(curNode == wx_add_b[curNodeIdx])
