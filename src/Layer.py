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
            for curNodeIdx, curNode in enumerate(self.var):
                if self.var_bounds_in["lb"][curNodeIdx] >= 0:
                    gmodel.addConstr(curNode == wx_add_b[curNodeIdx])
                elif self.var_bounds_in["ub"][curNodeIdx] <= 0:
                    gmodel.addConstr(curNode == 0)
                else:
                    # 1
                    gmodel.addConstr(curNode >= wx_add_b[curNodeIdx])

                    # 2
                    gmodel.addConstr(curNode <= wx_add_b[curNodeIdx] - self.var_bounds_in["lb"][curNodeIdx] * (1 - self.reluVar[curNodeIdx]))

                    # 3
                    gmodel.addConstr(curNode >= 0)

                    # 4
                    gmodel.addConstr(curNode <= self.var_bounds_in["ub"][curNodeIdx] * self.reluVar[curNodeIdx])
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
