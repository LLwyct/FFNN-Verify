import numpy as np
import gurobipy as gp
from Layer import ReluLayer, LinearLayer
from typing import List, Union
from gurobipy import GRB, quicksum
from networkClass import Network
import property
from ConstraintFormula import Disjunctive, Conjunctive


class Solver:
    def __init__(self, network: Network):
        self.m: gp.Model = gp.Model("ffnn")
        self.net: Network = network
        self.indexToVar: List[List] = []
        self.indexToReluvar: List[List] = []
        # 初始化网络级约束
        self.addNetworkConstraints()
        # 初始化人工约束,暂时用不到这个函数
        self.addManualConstraints()

    # 该函数用于添加输入层以及隐藏层中的网络约束，包括人为规定的输入层约束，人为规定的输出层约束不在该函数中添加
    def addNetworkConstraints(self):
        # 添加输入层的COUTINUE变量，并添加输入层约束
        self.indexToVar.append(
            [
                self.m.addVar(
                    ub=self.net.inputLmodel.var_bounds_out["ub"][i],
                    lb=self.net.inputLmodel.var_bounds_out["lb"][i],
                    vtype=GRB.CONTINUOUS,
                    name='X{}'.format(str(i))
                ) for i in range(self.net.inputLmodel.size)
            ]
        )
        self.net.inputLmodel.setVar(self.indexToVar[-1])
        self.m.update()

        # 添加隐藏层、输出层的CONTINUE变量,不包括input层
        for layer in self.net.lmodel:
            # 添加实数变量
            layer_var_bounds = layer.var_bounds_out
            if layer.type == "relu" or layer.type == "linear":
                ub, lb = None, None
                if layer_var_bounds["ub"] is None:
                    ub = [GRB.INFINITY for i in range(layer.size)]
                else:
                    ub = layer_var_bounds["ub"]
                if layer_var_bounds["lb"] is None:
                    lb = [-GRB.INFINITY for i in range(layer.size)]
                else:
                    lb = layer_var_bounds["lb"]

                if layer.type == "relu":
                    self.indexToVar.append(
                        [
                            self.m.addVar(
                                ub=ub[i],
                                lb=lb[i]
                            ) for i in range(layer.size)
                        ]
                    )
                '''
                当前默认的网络模式是input + n*relu + output
                因此当lmodel中出现linear层时，默认为输出层，因为lmodel不包括输入层，只能是输出层
                '''
                if layer.type == "linear":
                    self.indexToVar.append(
                        [
                            self.m.addVar(
                                ub=ub[i],
                                lb=lb[i],
                                name="Y{}".format(str(i))
                            ) for i in range(layer.size)
                        ]
                    )
            # 保存一份副本到lmodel的layer中
            layer.setVar(self.indexToVar[-1])
        self.m.update()


        # 添加relu节点变量，是二进制变量，0代表非激活y=0，1代表激活y=x
        for layer in self.net.lmodel:
            if layer.type == "relu":
                self.indexToReluvar.append([self.m.addVar(vtype=GRB.BINARY, name='reluVar') for _ in range(layer.size)])
                layer.setReluVar(self.indexToReluvar[-1])
            else:
                continue
        self.m.update()


        # 处理输入层到输出层的约束
        preLayer = self.net.inputLmodel
        for lidx, layer in enumerate(self.net.lmodel):
            # constrMethod=-1 表示使用全局的约束方法，为以后的优化做准备
            # constrMethod= 0 表示使用精确地混合整型编码方式进行约束
            # constrMethod= 1 表示使用三角松弛进行约束
            layer.addConstr(preLayer, self.m, constrMethod=-1)
            preLayer = layer

    def addManualConstraints(self):
        self.m.update()
        # 理论上在这里添加输出层上的约束
        constraints = property.acas_properties[self.net.propertyIndexReadyToVerify]["outputConstraints"][-1]
        if isinstance(constraints, Disjunctive):
            for constr in constraints.constraints:
                if constr[0] == "VarVar":
                    var1 = self.m.getVarByName(constr[1])
                    relation = constr[2]
                    var2 = self.m.getVarByName(constr[3])
                    if relation == "GT":
                        self.m.addConstr(var1 <= var2)
                    elif relation == "LT":
                        self.m.addConstr(var1 >= var2)
                    elif relation == "EQ":
                        pass
                    else:
                        raise Exception("输出约束关系异常")
                elif constr[0] == "VarValue":
                    var = self.m.getVarByName(constr[1])
                    relation = constr[2]
                    value = self.m.getVarByName(constr[3])
                    if relation == "GT":
                        self.m.addConstr(var <= value)
                    elif relation == "LT":
                        self.m.addConstr(var >= value)
                    elif relation == "EQ":
                        pass
                    else:
                        raise Exception("输出约束关系异常")
        elif isinstance(constraints, Conjunctive):
            constrlength = len(constraints.constraints)
            additionalConstrBinVar = [self.m.addVar(vtype=GRB.BINARY) for _ in range(constrlength)]
            for (i, constr) in enumerate(constraints.constraints):
                if constr[0] == "VarVar":
                    var1 = self.m.getVarByName(constr[1])
                    relation = constr[2]
                    var2 = self.m.getVarByName(constr[3])
                    if relation == "GT":
                        self.m.addConstr((additionalConstrBinVar[i] == 1) >> (var1 <= var2))
                    elif relation == "LT":
                        self.m.addConstr((additionalConstrBinVar[i] == 1) >> (var1 >= var2))
                    elif relation == "EQ":
                        pass
                    else:
                        raise Exception("输出约束关系异常")
                elif constr[0] == "VarValue":
                    var = self.m.getVarByName(constr[1])
                    relation = constr[2]
                    value = int(constr[3])
                    if relation == "GT":
                        self.m.addConstr((additionalConstrBinVar[i] == 1) >> (var <= value))
                    elif relation == "LT":
                        self.m.addConstr((additionalConstrBinVar[i] == 1) >> (var >= value))
                    elif relation == "EQ":
                        self.m.addConstr(var >= value)
                    else:
                        raise Exception("输出约束关系异常")
                self.m.update()
            self.m.addConstr(quicksum(additionalConstrBinVar) >= 1)
            self.m.addConstr(quicksum(additionalConstrBinVar) <= constrlength)
            self.m.update()


    def solve(self, verifyType):
        self.m.Params.outputFlag = 0
        self.m.optimize()
        if self.m.status == GRB.OPTIMAL:
            print(">>>>>>>>>>unsat>>>>>>>>>>")
            if verifyType == "acas":
                for i, X in enumerate(self.net.lmodel[-1].var):
                    print("y_" + str(i), X.x)
                for i, X in enumerate(self.net.inputLmodel.var):
                    print("X_" + str(i), X.x)
        else:
            print(">>>>>>>>>>>sat>>>>>>>>>>>")
