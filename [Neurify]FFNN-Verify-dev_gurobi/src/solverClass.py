import numpy as np
import gurobipy as gp
from Layer import ReluLayer, LinearLayer
from typing import List, Union
from gurobipy import GRB, quicksum
from networkClass import Network
import property
from ConstraintFormula import Disjunctive, Conjunctive
from options import GlobalSetting

class Solver:
    def __init__(self, network: Network):
        self.m: gp.Model = gp.Model("ffnn")
        self.net: Network = network
        self.indexToVar: List[List] = []
        self.indexToReluvar: List[List] = []

        while True and GlobalSetting.use_bounds_opt is True:
            self.indexToVar = []
            opt_num = self.optimize_bounds()
            print("optmise bounds number: ", opt_num)
            if opt_num == 0:
                break

        self.indexToVar = []
        # 初始化网络级约束
        self.addNetworkConstraints(self.m)
        # 初始化人工约束,暂时用不到这个函数
        self.addManualConstraints(self.m)

    # 该函数用于添加输入层以及隐藏层中的网络约束，包括人为规定的输入层约束，人为规定的输出层约束不在该函数中添加
    def addNetworkConstraints(self, m, optimize_bounds=False):
        # 添加输入层的COUTINUE变量，并添加输入层约束
        self.indexToVar.append(
            [
                m.addVar(
                    ub=self.net.inputLmodel.var_bounds_out["ub"][i],
                    lb=self.net.inputLmodel.var_bounds_out["lb"][i],
                    vtype=GRB.CONTINUOUS,
                    name='X{}'.format(str(i))
                ) for i in range(self.net.inputLmodel.size)
            ]
        )
        self.net.inputLmodel.setVar(self.indexToVar[-1])
        m.update()

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
                            m.addVar(
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
                            m.addVar(
                                ub=ub[i],
                                lb=lb[i],
                                name="Y{}".format(str(i))
                            ) for i in range(layer.size)
                        ]
                    )
            # 保存一份副本到lmodel的layer中
            layer.setVar(self.indexToVar[-1])
        m.update()


        if optimize_bounds == False:
            # 添加relu节点变量，是二进制变量，0代表非激活y=0，1代表激活y=x
            for layer in self.net.lmodel:
                if layer.type == "relu":
                    self.indexToReluvar.append([m.addVar(vtype=GRB.BINARY, name='reluVar') for _ in range(layer.size)])
                    layer.setReluVar(self.indexToReluvar[-1])
                else:
                    continue
            m.update()


        # 处理输入层到输出层的约束
        preLayer = self.net.inputLmodel
        for lidx, layer in enumerate(self.net.lmodel):
            # constrMethod=-1 表示使用全局的约束方法，为以后的优化做准备
            # constrMethod= 0 表示使用精确地混合整型编码方式进行约束
            # constrMethod= 1 表示使用三角松弛进行约束
            constrMethod = -1
            if optimize_bounds == True:
                constrMethod = 1
            layer.addConstr(preLayer, m, constrMethod=constrMethod)
            preLayer = layer

    def addManualConstraints(self, m):
        if self.net.verifyType == "mnist":
            return
        m.update()
        # 理论上在这里添加输出层上的约束
        constraints = property.acas_properties[self.net.propertyIndexReadyToVerify]["outputConstraints"][-1]
        if isinstance(constraints, Disjunctive):
            for constr in constraints.constraints:
                if constr[0] == "VarVar":
                    var1 = m.getVarByName(constr[1])
                    relation = constr[2]
                    var2 = m.getVarByName(constr[3])
                    if relation == "GT":
                        m.addConstr(var1 <= var2)
                    elif relation == "LT":
                        m.addConstr(var1 >= var2)
                    elif relation == "EQ":
                        pass
                    else:
                        raise Exception("输出约束关系异常")
                elif constr[0] == "VarValue":
                    var = m.getVarByName(constr[1])
                    relation = constr[2]
                    value = m.getVarByName(constr[3])
                    if relation == "GT":
                        m.addConstr(var <= value)
                    elif relation == "LT":
                        m.addConstr(var >= value)
                    elif relation == "EQ":
                        pass
                    else:
                        raise Exception("输出约束关系异常")
        elif isinstance(constraints, Conjunctive):
            constrlength = len(constraints.constraints)
            additionalConstrBinVar = [m.addVar(vtype=GRB.BINARY) for _ in range(constrlength)]
            for (i, constr) in enumerate(constraints.constraints):
                if constr[0] == "VarVar":
                    var1 = m.getVarByName(constr[1])
                    relation = constr[2]
                    var2 = m.getVarByName(constr[3])
                    if relation == "GT":
                        m.addConstr((additionalConstrBinVar[i] == 1) >> (var1 <= var2))
                    elif relation == "LT":
                        m.addConstr((additionalConstrBinVar[i] == 1) >> (var1 >= var2))
                    elif relation == "EQ":
                        pass
                    else:
                        raise Exception("输出约束关系异常")
                elif constr[0] == "VarValue":
                    var = m.getVarByName(constr[1])
                    relation = constr[2]
                    value = int(constr[3])
                    if relation == "GT":
                        m.addConstr((additionalConstrBinVar[i] == 1) >> (var <= value))
                    elif relation == "LT":
                        m.addConstr((additionalConstrBinVar[i] == 1) >> (var >= value))
                    elif relation == "EQ":
                        m.addConstr(var >= value)
                    else:
                        raise Exception("输出约束关系异常")
                m.update()
            m.addConstr(quicksum(additionalConstrBinVar) >= 1)
            m.addConstr(quicksum(additionalConstrBinVar) <= constrlength)
            m.update()

    def solve(self, verifyType):
        self.m.Params.outputFlag = 1
        self.m.optimize()
        res = ""
        if self.m.status == GRB.OPTIMAL:
            res= 'unsat'
            print(">>>>>>>>>>unsat>>>>>>>>>>")
            if verifyType == "acas":
                for i, X in enumerate(self.net.lmodel[-1].var):
                    print("y_" + str(i), X.x)
                for i, X in enumerate(self.net.inputLmodel.var):
                    print("X_" + str(i), X.x)
        else:
            res = 'sat'
            print(">>>>>>>>>>>sat>>>>>>>>>>>")
        return res

    def optimize_bounds(self):
        m = gp.Model("approx")
        m.Params.OutputFlag = 0
        opt_bounds_num = 0
        # 添加输入层的COUTINUE变量，并添加输入层约束
        '''
        我们所说的一个relu节点，它所对应的区间是out，relu节点与它的激活函数是绑定到一起的。
        而为什么在设计一个layer类的时候，要有var_bounds_in 和 var_bounds_out 两个区间呢？
        可以相像var_bounds_in像是神经元的一个突触，它用来接收上一层var_bounds_out和这一层的weight的点积而来的值
        这个值相当于relu节点的横坐标，当此值经过relu的激活后，才算的上是relu节点真正的值，并且此值作为out向下一层继续传递
        '''
        self.indexToVar.append(
            [
                m.addVar(
                    ub=self.net.inputLmodel.var_bounds_out["ub"][i],
                    lb=self.net.inputLmodel.var_bounds_out["lb"][i],
                    vtype=GRB.CONTINUOUS,
                    name='X{}'.format(str(i))
                ) for i in range(self.net.inputLmodel.size)
            ]
        )
        self.net.inputLmodel.setVar(self.indexToVar[-1])
        m.update()

        # 添加lmodel[0]的变量，因为对于第一个隐藏层来说它的边界由输入层经过一次区间传播而来，一定是精确的
        # 因此不需要对其边界进行优化
        self.indexToVar.append(
            [
                m.addVar(
                    ub=self.net.lmodel[0].var_bounds_out["ub"][i],
                    lb=self.net.lmodel[0].var_bounds_out["lb"][i],
                    vtype=GRB.CONTINUOUS,
                ) for i in range(self.net.lmodel[0].size)
            ]
        )
        self.net.lmodel[0].setVar(self.indexToVar[-1])
        m.update()

        self.net.lmodel[0].addConstr(self.net.inputLmodel, m, 1)

        # 添加lmodel[1]及其之后的变量，因为对于lmodel[0]，即使是区间算术，也很难造成较大的误差因此，不进行边界优化

        for i in range(1, self.net.layerNum-2):
            w = self.net.lmodel[i].weight
            x = self.net.lmodel[i - 1].var
            b = self.net.lmodel[i].bias
            wx_plus_b = np.dot(w, x) + b
            for j in range(self.net.lmodel[i].size):
                ub = self.net.lmodel[i].var_bounds_in["ub"][j]
                lb = self.net.lmodel[i].var_bounds_in["lb"][j]

                obj = m.addVar(
                    ub=ub,
                    lb=lb
                )
                # 这里只需要创建一个==的约束，因为这里优化的是激活前的上下界
                m.addConstr(obj == wx_plus_b[j])

                m.update()

                m.setObjective(obj, GRB.MINIMIZE)
                m.optimize()
                if m.status == GRB.Status.OPTIMAL:
                    lb = obj.X
                    if lb > self.net.lmodel[i].var_bounds_in["lb"][j]:
                        opt_bounds_num += 1
                        self.net.lmodel[i].var_bounds_in["lb"][j] = lb

                m.update()

                m.setObjective(obj, GRB.MAXIMIZE)
                m.optimize()
                if m.status == GRB.Status.OPTIMAL:
                    ub = obj.X
                    if ub < self.net.lmodel[i].var_bounds_in["ub"][j]:
                        opt_bounds_num += 1
                        self.net.lmodel[i].var_bounds_in["ub"][j] = ub

                m.remove(m.getVars()[-1])
                m.remove(m.getConstrs()[-1])
                m.update()
            # 当这层的每一个节点都分别优化完了， 更新了in的区间范围，别忘了更新out的区间范围
            self.net.lmodel[i].var_bounds_out["ub"] = np.maximum(self.net.lmodel[i].var_bounds_in["ub"], 0)
            self.net.lmodel[i].var_bounds_out["lb"] = np.maximum(self.net.lmodel[i].var_bounds_in["lb"], 0)

            # 当这层的每一个节点都分别优化完了，把这一层的全部节点添加到模型， 并建立与上一层的约束，进行下一轮循环
            self.indexToVar.append(
                [
                    m.addVar(
                        ub=self.net.lmodel[i].var_bounds_out["ub"][k],
                        lb=self.net.lmodel[i].var_bounds_out["lb"][k],
                        vtype=GRB.CONTINUOUS,
                    ) for k in range(self.net.lmodel[i].size)
                ]
            )
            self.net.lmodel[i].setVar(self.indexToVar[-1])
            self.net.lmodel[i].addConstr(self.net.lmodel[i-1], m, 1)
            m.update()

        return opt_bounds_num