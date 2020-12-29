# from mip import Model, BINARY, INTEGER, xsum
from networkClass import Network
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from Node import Node
from Layer import Layer


M = 1000000

class Solver:
    def __init__(self, network: Network, propertyFile: str):
        self.m: gp.Model = gp.Model("ffnn")
        self.net: Network = network
        self.indexToVar = []
        self.indexToReluvar = []
        self.indexToEpsilon = []
        self.propertyFile = propertyFile
        # 初始化网络级约束
        self.addNetworkConstraints()
        # 初始化人工约束
        # self.addManualConstraints()

    def addNetworkConstraints(self):
        # 添加输入层变量
        self.indexToVar.append(
            [
                self.m.addVar(
                    ub=self.net.inputLmodel.var_bounds_out["ub"][i],
                    lb=self.net.inputLmodel.var_bounds_out["lb"][i],
                    vtype=GRB.CONTINUOUS
                ) for i in range(self.net.inputLmodel.size)
            ]
        )
        self.net.inputLmodel.setVar(self.indexToVar[-1])
        self.m.update()



        # 添加非输入层变量 for all layer
        for layer in self.net.lmodel:
            assert isinstance(layer, Layer)
            # 添加实数变量
            layer_type = layer.type
            layer_var_bounds = layer.var_bounds_out
            if layer_type == "relu" or layer_type == "linear":
                ub, lb = None, None
                if layer_var_bounds["ub"] is None:
                    ub = [GRB.INFINITY for i in range(layer.size)]
                else:
                    ub = layer_var_bounds["ub"]
                if layer_var_bounds["lb"] is None:
                    lb = [-GRB.INFINITY for i in range(layer.size)]
                else:
                    lb = layer_var_bounds["lb"]
                self.indexToVar.append(
                    [
                        self.m.addVar(
                            ub=ub[i],
                            lb=lb[i]
                        ) for i in range(layer.size)
                    ]
                )
            # 保存一份副本到lmodel的layer中
            layer.setVar(self.indexToVar[-1])
        self.m.update()


        # 添加relu节点变量，是二进制变量，0代表非激活y=0，1代表激活y=x
        for layer in self.net.lmodel:
            layer_type = layer.type
            if layer_type == "relu":
                reluList = []
                for i in range(layer.size):
                    reluList.append(self.m.addVar(vtype=GRB.BINARY))
                self.indexToReluvar.append(reluList)
            else:
                continue
            layer.setReluVar(self.indexToReluvar[-1])
        self.m.update()

        # 处理输入层到输出层的约束
        preLayer = self.net.inputLmodel
        for lidx, layer in enumerate(self.net.lmodel):
            wx_add_b = np.dot(layer.weight, preLayer.var) + layer.bias

            for curNodeIdx, curNode in enumerate(layer.var):
                if layer.type == "linear":
                    self.m.addConstr(curNode == wx_add_b[curNodeIdx])
                elif layer.type == "relu":
                    if layer.var_bounds_in["lb"][curNodeIdx] >= 0:
                        self.m.addConstr(curNode == wx_add_b[curNodeIdx])
                    elif layer.var_bounds_in["ub"][curNodeIdx] <= 0:
                        self.m.addConstr(curNode == 0)
                    else:
                        # 1
                        self.m.addConstr(curNode >= wx_add_b[curNodeIdx])

                        # 2
                        # self.m.addConstr(curNode <= wx_add_b[curNodeIdx] + M * (1 - layer.reluVar[curNodeIdx]))
                        self.m.addConstr(curNode <= wx_add_b[curNodeIdx] - layer.var_bounds_in["lb"][curNodeIdx] * (1 - layer.reluVar[curNodeIdx]))

                        # 3
                        self.m.addConstr(curNode >= 0)

                        # 4
                        # self.m.addConstr(curNode <= M * layer.reluVar[curNodeIdx])
                        self.m.addConstr(curNode <= layer.var_bounds_in["ub"][curNodeIdx] * layer.reluVar[curNodeIdx])
            preLayer = layer

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
        # print('Obj: %g' % self.m.objVal)
        # if self.m == GRB.Status.Fes
        if self.m.status == GRB.OPTIMAL:
            print("unsat")
            for X in self.net.lmodel[-1].var:
                print("y_",X.x)
            for X in self.net.inputLmodel.var:
                print("X_",X.x)
        else:
            print(">>>>>>>>>>>sat")
