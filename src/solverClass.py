# from mip import Model, BINARY, INTEGER, xsum
from networkClass import Network
import numpy as np
import gurobipy as gp
from gurobipy import GRB

M = 100000

class Solver:
    def __init__(self, network: Network, propertyFile: str):
        self.m: gp.Model = gp.Model("ffnn")
        self.net: Network = network
        self.indexToVar = []
        self.indexToReluvar = []
        self.indexToEpsilon = []
        self.propertyFile = propertyFile
        # self.initEpsilonVar()
        # 初始化网络级约束
        self.addNetworkConstraints()
        # 初始化人工约束
        self.addManualConstraints()
        pass

    def initEpsilonVar(self):
        for i in range(self.net.layerNum):
            if i == 0:
                self.indexToEpsilon.append([])
            else:
                self.indexToEpsilon.append([self.m.addVar(lb=0, ub=0.000001) for i in range(self.net.eachLayerNums[i])])

    def addNetworkConstraints(self):
        # 添加输入层变量
        self.indexToVar.append(
            [
                self.m.addVar(
                    ub=self.net.inputLmodel.var_bounds["ub"][i],
                    lb=self.net.inputLmodel.var_bounds["lb"][i]
                ) for i in range(self.net.inputLmodel.size)
            ]
        )
        self.net.inputLmodel.setVar(self.indexToVar[self.indexToVar[-1]])
        self.m.update()


        # 添加非输入层变量 for all layer
        for layer in self.net.lmodel:
            # 添加实数变量
            layer_type = layer.type
            layer_var_bounds = layer.var_bounds
            if layer_type == "relu" or layer_type == "linear":
                if layer_var_bounds["ub"] is not None and layer_var_bounds["lb"] is not None:
                    self.indexToVar.append(
                        [
                            self.m.addVar(
                                ub=layer_var_bounds["ub"][i],
                                lb=layer_var_bounds["lb"][i]
                            ) for i in range(layer.size)
                        ]
                    )
                else:
                    self.indexToVar.append(
                        [
                            self.m.addVar() for i in range(layer.size)
                        ]
                    )
            # 保存一份副本到lmodel的layer中
            layer.setVar(self.indexToVar[-1])
        self.m.update()


        # 添加relu节点变量，是二进制变量，0代表非激活y=0，1代表激活y=x
        for layer in self.net.lmodel:
            # 第一层input不添加relu节点
            layer_type = layer.type
            if layer_type == "relu":
                self.indexToReluvar.append(
                    [
                        self.m.addVar(vtype=GRB.BINARY)
                                            for i in range(layer.size)
                    ]
                )
            else:
                continue
            layer.setReluVar(self.indexToReluvar[-1])
        self.m.update()


        # 对于每一个节点添加网络级线性约束
        # 遍历层
        for curLayerIdx, curLayer in enumerate(self.indexToVar):
            if curLayerIdx == 0:
                continue
            # 处理当前层curLayer的全部节点
            lastLayerIdx = curLayerIdx - 1
            lastLayerNodeNum = self.net.eachLayerNums[lastLayerIdx]
            if lastLayerNodeNum != len(self.indexToVar[lastLayerIdx]):
                raise
            # 先取出来bias，防止频繁访问net对象
            bias = self.net.biases[lastLayerIdx]
            weight = self.net.weights[lastLayerIdx]
            ans = np.dot(weight, self.indexToVar[lastLayerIdx])
            wx_add_b = ans + bias
            # 遍历层内的节点
            for curNodeIdx, curNode in enumerate(curLayer):
                # node is a var
                # 公式1
                # m += wx_add_b[curNodeIdx] - indexToEpsilon[curLayerIdx][curNodeIdx] <= curNode
                # m.addConstr(wx_add_b[curNodeIdx] - self.indexToEpsilon[curLayerIdx][curNodeIdx] <= curNode)
                self.m.addConstr(wx_add_b[curNodeIdx] <= curNode)

                # 公式2
                # m += wx_add_b[curNodeIdx] + M*(1 - indexToReluvar[curLayerIdx][curNodeIdx]) + indexToEpsilon[curLayerIdx][curNodeIdx] >= curNode
                # self.m.addConstr(wx_add_b[curNodeIdx] + M*(1 - self.indexToReluvar[curLayerIdx][curNodeIdx]) + self.indexToEpsilon[curLayerIdx][curNodeIdx] >= curNode)
                self.m.addConstr(wx_add_b[curNodeIdx] + M * (1 - self.indexToReluvar[curLayerIdx][curNodeIdx]) >= curNode)

                # 公式3
                # m += curNode >= 0
                self.m.addConstr(curNode >= 0)

                # 公式4
                # m += (curNode <= M * indexToReluvar[curLayerIdx][curNodeIdx])
                self.m.addConstr(curNode <= M * self.indexToReluvar[curLayerIdx][curNodeIdx])

    def addManualConstraints(self):
        # 添加输入层输出层上的约束
        inputConstraints, outputConstraints = self.loadProperty()
        finalLayerIndex = self.net.layerNum - 1
        for inputConstraint in inputConstraints:
            varIdx          = inputConstraint[0]
            equationType    = inputConstraint[1]
            scalar          = inputConstraint[2]
            if equationType == 0:
                # self.m += self.indexToVar[0][varIdx] == scalar
                self.m.addConstr(self.indexToVar[0][varIdx] == scalar)
                self.net.bounds[0][varIdx].lb = scalar
                self.net.bounds[0][varIdx].ub = scalar
            elif equationType == 1:
                # self.m += self.indexToVar[0][varIdx] <= scalar
                self.m.addConstr(self.indexToVar[0][varIdx] <= scalar)
                self.net.bounds[0][varIdx].ub = scalar
            elif equationType == 2:
                # self.m += self.indexToVar[0][varIdx] >= scalar
                self.m.addConstr(self.indexToVar[0][varIdx] >= scalar)
                self.net.bounds[0][varIdx].lb = scalar
        for outputConstraint in outputConstraints:
            varIdx          = outputConstraint[0]
            equationType    = outputConstraint[1]
            scalar          = outputConstraint[2]
            if equationType == 0:
                # self.m += self.indexToVar[finalLayerIndex][varIdx] == scalar
                self.m.addConstr(self.indexToVar[finalLayerIndex][varIdx] == scalar)
            elif equationType == 1:
                # self.m += self.indexToVar[finalLayerIndex][varIdx] <= scalar
                self.m.addConstr(self.indexToVar[finalLayerIndex][varIdx] <= scalar)
            elif equationType == 2:
                # self.m += self.indexToVar[finalLayerIndex][varIdx] >= scalar
                self.m.addConstr(self.indexToVar[finalLayerIndex][varIdx] >= scalar)

    def loadProperty(self):
        inputConstraints = []
        outputConstraints = []
        with open(self.propertyFile, "r") as f:
            line = f.readline()
            while line:
                if line == "":
                    raise Exception("IOError")
                var_equation_scala = line.split(" ")
                varName_varIndex = var_equation_scala[0].split("_")
                equation = var_equation_scala[1]
                scalar = var_equation_scala[2]
                varName = varName_varIndex[0]
                varIndex = varName_varIndex[1]
                if varName == "x" and equation == "==":
                    inputConstraints.append([int(varIndex), 0, float(scalar)])
                elif varName == "x" and equation == "<=":
                    inputConstraints.append([int(varIndex), 1, float(scalar)])
                elif varName == "x" and equation == ">=":
                    inputConstraints.append([int(varIndex), 2, float(scalar)])
                elif varName == "y" and equation == "==":
                    outputConstraints.append([int(varIndex), 0, float(scalar)])
                elif varName == "y" and equation == "<=":
                    outputConstraints.append([int(varIndex), 1, float(scalar)])
                elif varName == "y" and equation == ">=":
                    outputConstraints.append([int(varIndex), 2, float(scalar)])
                else:
                    raise Exception("IOError")
                line = f.readline()
        return inputConstraints, outputConstraints

    def solve(self):
        self.m.optimize()
        # if self.m.num_solutions:
        #     print("-----------------solutation found!-----------------")
        #     for i in range(self.net.layerNum):
        #         for j, node in enumerate(self.indexToVar[i]):
        #             print("x_{}{}:{}".format(i, j, node.x))
        #
        #     for i in range(self.net.layerNum):
        #         for j, node in enumerate(self.indexToReluvar[i]):
        #             print("&_{}{}:{}".format(i, j, node.x))
        #     print("-----------------solutation found!-----------------")
        # else:
        #     print("-----------------solutation not found!-----------------")
        #     print("unsat")
        #     print("-----------------solutation not found!-----------------")
        print('Obj: %g' % self.m.objVal)