import sys
import numpy as np
from numpy import ndarray
from typing import Union, List, Dict, Optional
from gurobipy import Model
from options import GlobalSetting
from LinearFunction import LinearFunction

class Layer:
    def __init__(self, id, layer_type="unknown"):
        # "linear" | "relu" | "input" | "unknown"
        self.type: str = layer_type
        self.size: Optional[int] = None
        self.id = id
        self.var_bounds_in: Dict[str, Optional['ndarray']] = {
            "ub": None,
            "lb": None
        }
        self.var_bounds_out: Dict[str, Optional[List]] = {
            "ub": None,
            "lb": None
        }
        self.var_bounds_in_cmp: Dict[str, Optional['ndarray']] = {
            "ub": None,
            "lb": None
        }
        self.var_bounds_out_cmp: Dict[str, Optional[List]] = {
            "ub": None,
            "lb": None
        }
        self.bound_equations: Dict[str, Dict[str, Optional['LinearFunction']]] = {
            'in': {
                'lb': None,
                'ub': None
            },
            'out': {
                'lb': None,
                'ub': None
            }
        }
        self.bound_equations_cmp: Dict[str, Dict[str, Optional['LinearFunction']]] = {
            'in': {
                'lb': None,
                'ub': None
            },
            'out': {
                'lb': None,
                'ub': None
            }
        }
        self.var = None
        self.nodeList = []

    def setVar(self, varlist):
        self.var = np.array(varlist)

    def getVar(self):
        if self.var is not None:
            return self.var
        else:
            pass

    def setSize(self, size):
        self.size = size

    def _compute_in_bounds_sia_Eqs(self, pLayerUpperEq: Optional['LinearFunction'], pLayerLowerEq: Optional['LinearFunction']) -> Dict[str, 'LinearFunction']:
        weight_plus = np.maximum(self.weight, np.zeros(self.weight.shape))
        weight_neg = np.minimum(self.weight, np.zeros(self.weight.shape))

        pUpperMatrix = pLayerUpperEq.matrix
        pLowerMatrix = pLayerLowerEq.matrix
        upperMatrix = weight_plus.dot(pUpperMatrix) + weight_neg.dot(pLowerMatrix)
        lowerMatrix = weight_plus.dot(pLowerMatrix) + weight_neg.dot(pUpperMatrix)

        pUpperConst = pLayerUpperEq.offset
        pLowerConst = pLayerLowerEq.offset
        upperConst = weight_plus.dot(pUpperConst) + weight_neg.dot(pLowerConst) + self.bias
        lowerConst = weight_plus.dot(pLowerConst) + weight_neg.dot(pUpperConst) + self.bias

        return {
            "ub": LinearFunction(upperMatrix, upperConst),
            "lb": LinearFunction(lowerMatrix, lowerConst)
        }

    def _compute_out_bounds_0sia_or_1slr_Eqs(self, Eqin, inputLayer, method):
        '''
        return {
            "ub": Eqin["ub"].getUpperOutEqThroughRelu(inputLayer),
            "lb": Eqin["lb"].getLowerOutEqThroughRelu(inputLayer)
        }
        '''
        inUPEq: 'LinearFunction' = Eqin["ub"]
        inLOWEq: 'LinearFunction' = Eqin["lb"]

        # 先获得一份未来用于构造函数参数的拷贝
        newUpMatrix = inUPEq.matrix.copy()
        newUpOffset = inUPEq.offset.copy()

        newLowMatrix = inLOWEq.matrix.copy()
        newLowOffset = inLOWEq.offset.copy()

        # 对UpEq分别调用MaxBoundsValue和MinBoundsValue
        # 这个和main的part2的1式相同，是上界的上界
        UPEqUpper = inUPEq.computeMaxBoundsValue(inputLayer)
        # 是上界的下界
        UPEqLower = inUPEq.computeMinBoundsValue(inputLayer)

        # 对LowEq分别调用MaxBoundsValue和MinBoundsValue
        # 是下界的上界
        LOWEqUpper = inLOWEq.computeMaxBoundsValue(inputLayer)
        # 这个和main的part2的2式相同，是下界的下界
        LOWEqLower = inLOWEq.computeMinBoundsValue(inputLayer)

        # 0:sia 1:slr
        if method == 0:
            for i in range(self.size):
                if UPEqUpper[i] <= 0:
                    # 如果上界的上界都小于0了，那么出边的Eq的上下界都为0
                    newUpMatrix[i,:] = 0
                    newUpOffset[i] = 0
                    newLowMatrix[i, :] = 0
                    newLowOffset[i] = 0
                elif LOWEqLower[i] >= 0:
                    # 否则什么都不做
                    continue
                else:
                    #lastMatrix = newUpMatrix[i,:][-1]
                    #lastOffset = newUpOffset[i]
                    # 如果处于位置状态，首先下界的Eq置为0
                    newLowMatrix[i,:] = 0
                    newLowOffset[i] = 0
                    # 上界按情况判断
                    if UPEqLower[i] < 0:
                        # 如果上界有可能小于0
                        newUpMatrix[i, :] = 0
                        newUpOffset[i] = UPEqUpper[i]
        # slr
        elif method == 1:
            for i in range(self.size):
                if UPEqUpper[i] <= 0:
                    # 如果上界的上界都小于0了，那么出边的Eq的上下界都为0
                    newUpMatrix[i, :] = +0
                    newUpOffset[i] = +0
                elif UPEqLower[i] >= 0:
                    # activate node 否则什么都不做
                    continue
                else:
                    adj = UPEqUpper[i] / (UPEqUpper[i] - UPEqLower[i])
                    newUpMatrix[i, :] = adj * newUpMatrix[i, :]
                    newUpOffset[i] = adj * newUpOffset[i] - adj * UPEqLower[i]

                if LOWEqUpper[i] <= 0:
                    newLowMatrix[i, :] = +0
                    newLowOffset[i] = +0
                elif LOWEqLower[i] >= 0:
                    continue
                else:
                    adj = LOWEqUpper[i] / (LOWEqUpper[i] - LOWEqLower[i])
                    newLowMatrix[i, :] = adj * newLowMatrix[i, :]
                    newLowOffset[i] = adj * newLowOffset[i]

        return {
            "ub": LinearFunction(newUpMatrix, newUpOffset),
            "lb": LinearFunction(newLowMatrix, newLowOffset)
        }


class InputLayer(Layer):
    def __init__(self, id, layer_type="input", size=0):
        super(InputLayer, self).__init__(id, layer_type)
        self.type = layer_type
        self.size = size

    def setBounds(self, ub: 'ndarray', lb: 'ndarray'):
        self.var_bounds_in["ub"] = self.var_bounds_out["ub"] = ub
        self.var_bounds_in["lb"] = self.var_bounds_out["lb"] = lb

class ReluLayer(Layer):
    def __init__(self, id, w: 'ndarray', b: 'ndarray', layer_type: str = "relu"):
        super(ReluLayer, self).__init__(id, layer_type)
        self.type:      str     = layer_type
        self.size:      int     = b.size
        self.weight:    'ndarray' = w
        self.bias:      'ndarray' = b
        self.reluVar:   'ndarray' = np.empty(self.size)

    def addConstr(self, preLayer: 'Layer', gmodel: 'Model', constrMethod: int):
        wx_add_b = np.dot(self.weight, preLayer.var) + self.bias
        constrMethod = GlobalSetting.constrMethod if constrMethod == -1 else constrMethod
        if constrMethod == 0:
            '''
            0代表使用带激活变量的精确约束MILP
            1代表使用三角松弛
            '''
            # 这个变量用于计算各种边界预处理方法（区间传播、符号传播、符号线性松弛等）所带来的边界束紧，对于每一层分别减少了多少binary变量
            ignoreBinaryVarNum = 0
            # 用这两个变量来测试并打印在区间传播的过程中，每一隐藏层Relu节点输入的上界的最大值和下界的最小值，从而判断自定义的M是否满足要求。
            maxUpper = -1 * sys.maxsize
            minLower = +1 * sys.maxsize
            sum_diff = 0
            for curNodeIdx, curNode in enumerate(self.var):
                # 统计每层中出现的最大上界和最小下界
                if self.var_bounds_in["ub"][curNodeIdx] > maxUpper:
                    maxUpper = self.var_bounds_in["ub"][curNodeIdx]
                if self.var_bounds_in["lb"][curNodeIdx] < minLower:
                    minLower = self.var_bounds_in["lb"][curNodeIdx]

                # 统计上下界的差值
                sum_diff += self.var_bounds_in["ub"][curNodeIdx] - self.var_bounds_in["lb"][curNodeIdx]

                if self.var_bounds_in["lb"][curNodeIdx] >= 0:
                    gmodel.addConstr(curNode == wx_add_b[curNodeIdx])
                    ignoreBinaryVarNum += 1
                elif self.var_bounds_in["ub"][curNodeIdx] <= 0:
                    gmodel.addConstr(curNode == 0)
                    ignoreBinaryVarNum += 1
                else:
                    upper_bounds = self.var_bounds_in["ub"][curNodeIdx]
                    lower_bounds = self.var_bounds_in["lb"][curNodeIdx]
                    # 1
                    gmodel.addConstr(curNode >= wx_add_b[curNodeIdx])

                    # 2
                    gmodel.addConstr(curNode <= wx_add_b[curNodeIdx] - lower_bounds * (1 - self.reluVar[curNodeIdx]))

                    # 3
                    gmodel.addConstr(curNode >= 0)

                    # 4
                    gmodel.addConstr(curNode <= upper_bounds * self.reluVar[curNodeIdx])
            '''print()
            print(self.id, self.type, self.size, ignoreBinaryVarNum)
            print('maxupper', maxUpper)
            print('minLower', minLower)
            print('sum_diff', sum_diff)'''

        elif constrMethod == 1:
            '''
            0代表使用带激活变量的精确约束MILP
            1代表使用三角松弛
            '''
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

    # 该函数用于计算在符号传播或符号线性松弛等预计算过程中每一层的Eqution以及具体边界的传播情况
    def compute_Eq_and_bounds_0sia_or_1slr(self, preLayer: 'Layer', inputLayer: 'InputLayer', method: int):
        # 主函数，用于计算符号传播
        # part1：计算当前层输入的Eq
        self.bound_equations["in"] = self._compute_in_bounds_sia_Eqs(
            preLayer.bound_equations["out"]["ub"],
            preLayer.bound_equations["out"]["lb"]
        )

        # part2：
        # 计算当前层输入的真实边界的值，由Eq和input决定，不由上一层决定
        # 注意这里是对上界EQ调用maxValue，对下界EQ调用minValue
        self.var_bounds_in["ub"] = self.bound_equations["in"]["ub"].computeMaxBoundsValue(inputLayer)
        self.var_bounds_in["lb"] = self.bound_equations["in"]["lb"].computeMinBoundsValue(inputLayer)


        # part3：计算输出的Eq，与part1对应
        self.bound_equations["out"] = self._compute_out_bounds_0sia_or_1slr_Eqs(
            self.bound_equations["in"],
            inputLayer,
            method
        )

        # part4：
        # 计算当前层输出的真实边界的值，由Eq和input决定，不由上一层决定
        # 注意这里是对上界EQ调用maxValue，对下界EQ调用minValue
        self.var_bounds_out["ub"] = self.bound_equations["out"]["ub"].computeMaxBoundsValue(inputLayer)
        self.var_bounds_out["lb"] = self.bound_equations["out"]["lb"].computeMinBoundsValue(inputLayer)

        self.var_bounds_out['ub'] = np.maximum(self.var_bounds_out['ub'], 0)
        self.var_bounds_out['lb'] = np.maximum(self.var_bounds_out['lb'], 0)
        self.computeBoundsError()

    # 该函数用于结合计算
    def compute_Eq_and_bounds_sia_and_slr(self, preLayer: 'Layer', inputLayer: 'InputLayer'):
        # 主函数，用于计算符号传播
        # part1：计算当前层输入的Eq
        self.bound_equations["in"] = self._compute_in_bounds_sia_Eqs(
            preLayer.bound_equations["out"]["ub"],
            preLayer.bound_equations["out"]["lb"]
        )

        self.bound_equations_cmp["in"] = self._compute_in_bounds_sia_Eqs(
            preLayer.bound_equations_cmp["out"]["ub"],
            preLayer.bound_equations_cmp["out"]["lb"]
        )
        # part2：
        # 计算当前层输入的真实边界的值，由Eq和input决定，不由上一层决定
        # 注意这里是对上界EQ调用maxValue，对下界EQ调用minValue
        self.var_bounds_in["ub"] = self.bound_equations["in"]["ub"].computeMaxBoundsValue(inputLayer)
        self.var_bounds_in["lb"] = self.bound_equations["in"]["lb"].computeMinBoundsValue(inputLayer)

        self.var_bounds_in_cmp = {
            "ub": self.bound_equations_cmp["in"]["ub"].computeMaxBoundsValue(inputLayer),
            "lb": self.bound_equations_cmp["in"]["lb"].computeMinBoundsValue(inputLayer)
        }

        # part3：计算输出的Eq，与part1对应
        self.bound_equations["out"] = self._compute_out_bounds_0sia_or_1slr_Eqs(
            self.bound_equations["in"],
            inputLayer,
            1
        )

        self.bound_equations_cmp["out"] = self._compute_out_bounds_0sia_or_1slr_Eqs(
            self.bound_equations_cmp["in"],
            inputLayer,
            0
        )

        # part4：
        # 计算当前层输出的真实边界的值，由Eq和input决定，不由上一层决定
        # 注意这里是对上界EQ调用maxValue，对下界EQ调用minValue
        self.var_bounds_out["ub"] = self.bound_equations["out"]["ub"].computeMaxBoundsValue(inputLayer)
        self.var_bounds_out["lb"] = self.bound_equations["out"]["lb"].computeMinBoundsValue(inputLayer)

        self.var_bounds_out_cmp = {
            "ub": self.bound_equations_cmp["in"]["ub"].computeMaxBoundsValue(inputLayer),
            "lb": self.bound_equations_cmp["in"]["lb"].computeMinBoundsValue(inputLayer)
        }

        self.var_bounds_out['ub'] = np.maximum(self.var_bounds_out['ub'], 0)
        self.var_bounds_out['lb'] = np.maximum(self.var_bounds_out['lb'], 0)
        self.var_bounds_out_cmp['ub'] = np.maximum(self.var_bounds_out_cmp['ub'] , 0)
        self.var_bounds_out_cmp['lb'] = np.maximum(self.var_bounds_out_cmp['lb'] , 0)

        '''
        for i in range(self.size):
            if self.var_bounds_in_cmp["ub"][i] < self.var_bounds_in["ub"][i]:
                self.var_bounds_in["ub"][i] = self.var_bounds_in_cmp["ub"][i]
                print('trigger')
            if self.var_bounds_in_cmp["lb"][i] > self.var_bounds_in["lb"][i]:
                self.var_bounds_in["lb"][i] = self.var_bounds_in_cmp["lb"][i]
                print('trigger')

            if self.var_bounds_out_cmp["ub"][i] < self.var_bounds_out["ub"][i]:
                self.var_bounds_out["ub"][i] = self.var_bounds_out_cmp["ub"][i]
            if self.var_bounds_out_cmp["lb"][i] > self.var_bounds_out["lb"][i]:
                self.var_bounds_out["lb"][i] = self.var_bounds_out_cmp["lb"][i]
        '''

        self.computeBoundsError()

    # 计算在符号传播或符号线性松弛等预计算过程中是否出现下界大于上界的情况
    def computeBoundsError(self):
        num = 0
        for i in range(self.size):
            if self.var_bounds_in["lb"][i] > self.var_bounds_in["ub"][i]:
                num += 1
                #print("Out of in bounds ", i)
            if self.var_bounds_in_cmp["lb"] is not None and self.var_bounds_in_cmp["lb"][i] > self.var_bounds_in_cmp["ub"][i]:
                num += 1
                #print("optimize lead Out of inner bounds ", i)
        if num != 0:
            pass
            #print("value Error , inbounds, ", self.id, num)

        num = 0
        for i in range(self.size):
            if self.var_bounds_out["lb"][i] > self.var_bounds_out["ub"][i]:
                num += 1
                #print("Out of out bounds ", i)
            if self.var_bounds_out_cmp["lb"] is not None and self.var_bounds_out_cmp["lb"][i] > self.var_bounds_out_cmp["ub"][i]:
                num += 1
                #print("optimize lead out of outter bounds ", i)
        if num != 0:
            pass
            #print("value Error , outbounds, ", self.id, num)

    def getFixedNodeNum(self):
        fixedNodeNum = self.size
        for i in range(self.size):
            if self.var_bounds_in["ub"][i] > 0 and self.var_bounds_in["lb"][i] < 0:
                fixedNodeNum -= 1
        if fixedNodeNum < 0:
            raise Exception("Error")
        return fixedNodeNum

    def getNotFixedNode(self):
        for i in range(self.size):
            if self.var_bounds_in["ub"] < 0 or self.var_bounds_in["lb"] > 0:
                return self.var[i]

class LinearLayer(Layer):
    def __init__(self, id, w:ndarray, b:ndarray, layer_type: str = "linear"):
        super(LinearLayer, self).__init__(id, layer_type)
        self.type:      str     = layer_type
        self.size:      int     = b.size
        self.weight:    ndarray = w
        self.bias:      ndarray = b

    def addConstr(self, preLayer, gmodel: Model, constrMethod):
        if GlobalSetting.constrMethod == 0 or GlobalSetting.constrMethod == 1:
            wx_add_b = np.dot(self.weight, preLayer.var) + self.bias
            for curNodeIdx, curNode in enumerate(self.var):
                gmodel.addConstr(curNode == wx_add_b[curNodeIdx])
        maxUpper = -1 * sys.maxsize
        minLower = +1 * sys.maxsize
        sum_diff = 0
        for i in range(self.size):
            if self.var_bounds_in["ub"][i] > maxUpper:
                maxUpper = self.var_bounds_in["ub"][i]
            if self.var_bounds_in["lb"][i] < minLower:
                minLower = self.var_bounds_in["lb"][i]
            sum_diff += self.var_bounds_in["ub"][i] - self.var_bounds_in["lb"][i]
        #print(self.id, self.type)
        #print('maxupper', maxUpper)
        #print('minLower', minLower)
        #print('sum_diff', sum_diff)

    def compute_Eq_and_bounds(self, preLayer, inputLayer):
        # 这里因为对于Linear层，in和out是一样的，所以我们直接使用in的函数结果返回给out，因为在利用区间的值赋值给M的时候，只需要out不需要in
        self.bound_equations["in"] = self.bound_equations["out"] = self._compute_in_bounds_sia_Eqs(
            preLayer.bound_equations["out"]["ub"],
            preLayer.bound_equations["out"]["lb"]
        )
        self.var_bounds_in["ub"] = self.var_bounds_out["ub"] = self.bound_equations["out"]["ub"].computeMaxBoundsValue(inputLayer)
        self.var_bounds_in["lb"] = self.var_bounds_out["lb"] = self.bound_equations["out"]["lb"].computeMinBoundsValue(inputLayer)

        if preLayer.bound_equations_cmp["in"]["ub"] is not None:
            self.bound_equations_cmp["in"] = self.bound_equations_cmp["out"] = self._compute_in_bounds_sia_Eqs(
                preLayer.bound_equations_cmp["out"]["ub"],
                preLayer.bound_equations_cmp["out"]["lb"]
            )
            self.var_bounds_in_cmp["ub"] = self.var_bounds_out_cmp["ub"] = self.bound_equations_cmp["out"]["ub"].computeMaxBoundsValue(inputLayer)
            self.var_bounds_in_cmp["lb"] = self.var_bounds_out_cmp["lb"] = self.bound_equations_cmp["out"]["lb"].computeMinBoundsValue(inputLayer)


class OutputLayer(LinearLayer):
    def __init__(self, id: int, w:'ndarray', b:'ndarray', layer_type: str="output", size=0):
        super(OutputLayer, self).__init__(id, w, b)
        self.type = layer_type
        self.size = size
