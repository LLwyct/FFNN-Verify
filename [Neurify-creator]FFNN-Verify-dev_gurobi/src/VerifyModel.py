import copy
from typing import List
import numpy as np
import sys
from ConstraintFormula import Conjunctive, Disjunctive
from Specification import Specification
from options import GlobalSetting
from Layer import ReluLayer, LinearLayer
from LinearFunction import LinearFunction
from LayerModel import LayerModel
import gurobipy as gp
from gurobipy import GRB, quicksum


class VerifyModel:
    def __init__(self, id: str, lmodel: 'LayerModel', spec: 'Specification'):
        self.id: str = id
        self.lmodel: 'LayerModel' = copy.deepcopy(lmodel)
        self.spec: 'Specification' = copy.deepcopy(spec)
        self.verifyType = "acas"
        self.lmodel.loadSpec(self.spec)
        self.gmodel = None
        self.indexToVar: List[List] = []
        self.indexToReluvar: List[List] = []

    def verify(self) -> bool:
        '''
        当sat满足约束的取反时返回False
        当unsat不满足约束的取反时返回True
        '''
        self.gmodel = gp.Model()
        self.gmodel.Params.OutputFlag = 0
        self.indexToVar = []
        self.indexToReluvar: List[List] = []
        while True and GlobalSetting.use_bounds_opt is True:
            # 属于预处理的步骤，使用三角近似来优化边界，要在initAllBounds之后执行
            self.indexToVar = []
            opt_num = self.optimize_bounds()
            print("optmise bounds number: ", opt_num)
            if opt_num == 0:
                break
        if GlobalSetting.use_binary_heuristic_method == 0:
            self.addGurobiVarFromLayerModel(self.gmodel)
            self.addGurobiVarFromLayerModel(self.gmodel)
            self.addManualConstraints(self.gmodel)
            self.gmodel.optimize()
            if self.gmodel.status == GRB.OPTIMAL:
                # print("launch slover", self.id, False)
                return False
            else:
                # print("launch slover", self.id, True)
                return True
        elif GlobalSetting.use_binary_heuristic_method == 1:
            self.binaryHeuristicAddConstrs()


    def binaryHeuristicAddConstrs(self):
        # init 到初始状态
        self.gmodel.remove(self.gmodel.getVars())
        self.gmodel.remove(self.gmodel.getConstrs())
        self.indexToVar = []
        self.indexToReluvar: List[List] = []
        notFixedNode = []
        for layer in self.lmodel.lmodels:
            if layer.type == "relu":
                layer.getNotFixedNode()
        lo = 0
        hi = self.lmodel.layerNum - 2
        while lo <= hi:

        self.addGurobiVarFromLayerModel(self.gmodel)
        self.addManualConstraints(self.gmodel)

    def initAllBounds(self):
        # 初始化/预处理隐藏层及输出层的边界
        # 0 MILP with bigM
        # 1 MILP with ia  区间传播
        # 2 MILP with sia 符号区间传播
        # 3 MILP with slr 符号线性松弛
        if GlobalSetting.preSolveMethod == 0:
            pass
        elif GlobalSetting.preSolveMethod == 1:
            self.intervalPropation()
        elif GlobalSetting.preSolveMethod == 2:
            self.symbolIntervalPropation_0sia_or_1slr(0)
        elif GlobalSetting.preSolveMethod == 3:
            self.symbolIntervalPropation_0sia_or_1slr(1)
        elif GlobalSetting.preSolveMethod == 4:
            res = self.spec.getInputBounds()
            self.lmodel.inputLayer.var_bounds_in_cmp["ub"] = self.lmodel.inputLayer.var_bounds_out_cmp["ub"] = res[0]
            self.lmodel.inputLayer.var_bounds_in_cmp["lb"] = self.lmodel.inputLayer.var_bounds_out_cmp["lb"] = res[1]
            self.symbolIntervalPropation_sia_and_slr()

    def intervalPropation(self):
        preLayer_u = self.lmodel.inputLayer.var_bounds_out["ub"]
        preLayer_l = self.lmodel.inputLayer.var_bounds_out["lb"]
        for layer in self.lmodel.lmodels:
            w_active = np.maximum(layer.weight, np.zeros(layer.weight.shape))
            w_inactive = np.minimum(layer.weight, np.zeros(layer.weight.shape))
            # l_hat = w_+ * l_{i-1} + w_- * u_{i-1}
            l_left = np.dot(w_active, preLayer_l)
            l_right = np.dot(w_inactive, preLayer_u)
            l_hat = l_left + l_right
            # u_hat = w_+ * u_{i-1} + w_- * l_{i-1}
            u_left = np.dot(w_active, preLayer_u)
            u_right = np.dot(w_inactive, preLayer_l)
            u_hat = u_left + u_right
            layer.var_bounds_in["ub"] = u_hat + layer.bias
            layer.var_bounds_in["lb"] = l_hat + layer.bias
            if isinstance(layer, ReluLayer):
                # 当当前层为ReLU时，out的边界要在in的基础上经过激活函数处理，注意max函数和maximum的区别
                # max函数是在axis上取最大值，maximum是把arg1向量中的每一个数和对应的arg2向量中的每个数取最大值。
                preLayer_u = layer.var_bounds_out["ub"] = np.maximum(
                    layer.var_bounds_in["ub"], np.zeros(u_hat.shape))
                preLayer_l = layer.var_bounds_out["lb"] = np.maximum(
                    layer.var_bounds_in["lb"], np.zeros(l_hat.shape))
            elif isinstance(layer, LinearLayer):
                # 如果当前层是linear层，则不需要经过ReLU激活函数
                preLayer_u = layer.var_bounds_out["ub"] = layer.var_bounds_in["ub"]
                preLayer_l = layer.var_bounds_out["lb"] = layer.var_bounds_in["lb"]
            for i in range(len(layer.var_bounds_in["ub"])):
                if layer.var_bounds_in["ub"][i] < layer.var_bounds_in["lb"][i]:
                    raise Exception

    def symbolIntervalPropation_0sia_or_1slr(self, method):
        inputLayerSize = self.lmodel.inputLayer.size
        self.lmodel.inputLayer.bound_equations["out"]["lb"] = LinearFunction(
            np.identity(inputLayerSize), np.zeros(inputLayerSize))
        self.lmodel.inputLayer.bound_equations["out"]["ub"] = LinearFunction(
            np.identity(inputLayerSize), np.zeros(inputLayerSize))

        preLayer = self.lmodel.inputLayer
        for layer in self.lmodel.lmodels:
            if layer.type == "relu":
                assert isinstance(layer, ReluLayer)
                layer.compute_Eq_and_bounds_0sia_or_1slr(preLayer, self.lmodel.inputLayer, method)
            elif layer.type == "linear":
                assert isinstance(layer, LinearLayer)
                layer.compute_Eq_and_bounds(preLayer, self.lmodel.inputLayer)
            preLayer = layer

    def symbolIntervalPropation_sia_and_slr(self):
        inputLayerSize = self.lmodel.inputLayer.size
        self.lmodel.inputLayer.bound_equations["out"]["lb"] = LinearFunction(np.identity(inputLayerSize),
                                                                        np.zeros(inputLayerSize))
        self.lmodel.inputLayer.bound_equations["out"]["ub"] = LinearFunction(np.identity(inputLayerSize),
                                                                        np.zeros(inputLayerSize))

        self.lmodel.inputLayer.bound_equations_cmp["out"]["lb"] = LinearFunction(np.identity(inputLayerSize),
                                                                            np.zeros(inputLayerSize))
        self.lmodel.inputLayer.bound_equations_cmp["out"]["ub"] = LinearFunction(np.identity(inputLayerSize),
                                                                            np.zeros(inputLayerSize))
        preLayer = self.lmodel.inputLayer
        for layer in self.lmodel.lmodels:
            if layer.type == "relu":
                assert isinstance(layer, ReluLayer)
                layer.compute_Eq_and_bounds_sia_and_slr(
                    preLayer, self.lmodel.inputLayer)
            elif layer.type == "linear":
                assert isinstance(layer, LinearLayer)
                layer.compute_Eq_and_bounds(preLayer, self.lmodel.inputLayer)
            preLayer = layer

        for layer in self.lmodel.lmodels:
            # slr_out_diff = 0
            # merge_out_diff = 0
            maxUpper_out_sia = -1 * sys.maxsize
            minLower_out_sia = +1 * sys.maxsize
            maxUpper_out_slr = -1 * sys.maxsize
            minLower_out_slr = +1 * sys.maxsize
            for i in range(layer.size):
                # slr_out_diff += layer.var_bounds_out["ub"][i] - \
                    # layer.var_bounds_out["lb"][i]
                if layer.var_bounds_out_cmp["ub"][i] > maxUpper_out_sia:
                    maxUpper_out_sia = layer.var_bounds_out_cmp["ub"][i]
                if layer.var_bounds_out_cmp["lb"][i] < minLower_out_sia:
                    minLower_out_sia = layer.var_bounds_out_cmp["lb"][i]
                if layer.var_bounds_out["ub"][i] > maxUpper_out_slr:
                    maxUpper_out_slr = layer.var_bounds_out["ub"][i]
                if layer.var_bounds_out["lb"][i] < minLower_out_slr:
                    minLower_out_slr = layer.var_bounds_out["lb"][i]
            # print(layer.id, "outter")
            # print(maxUpper_out_sia, maxUpper_out_slr)
            if layer.id < self.lmodel.layerNum - 1:
                layer.var_bounds_in["ub"] = np.minimum(
                    layer.var_bounds_in["ub"], layer.var_bounds_in_cmp["ub"])
                layer.var_bounds_in["lb"] = np.maximum(
                    layer.var_bounds_in["lb"], layer.var_bounds_in_cmp["lb"])
                #layer.var_bounds_out["ub"] = np.minimum(layer.var_bounds_out["ub"], layer.var_bounds_out_cmp["ub"])
                #layer.var_bounds_out["lb"] = np.maximum(layer.var_bounds_out["lb"], layer.var_bounds_out_cmp["lb"])

            # for i in range(layer.size):
            #     merge_out_diff += layer.var_bounds_out["ub"][i] - layer.var_bounds_out["lb"][i]
            # print(slr_out_diff, merge_out_diff)

    def getFixedNodeRatio(self) -> float:
        return self.lmodel.getFixedNodeRatio()

    def addGurobiVarFromLayerModel(self, m: 'gp.Model'):
        # 添加输入层的COUTINUE变量，并添加输入层约束
        self.indexToVar.append(
            [
                m.addVar(
                    ub=self.lmodel.inputLayer.var_bounds_out["ub"][i],
                    lb=self.lmodel.inputLayer.var_bounds_out["lb"][i],
                    vtype=GRB.CONTINUOUS,
                    name='X{}'.format(str(i))
                ) for i in range(self.lmodel.inputLayer.size)
            ]
        )
        self.lmodel.inputLayer.setVar(self.indexToVar[-1])
        m.update()

        # 添加隐藏层、输出层的CONTINUE变量,不包括input层
        for layer in self.lmodel.lmodels:
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

        '''
        这里不知道为什么有这一行代码，预计要删除
        if optimize_bounds == False:
            # 添加relu节点变量，是二进制变量，0代表非激活y=0，1代表激活y=x
            for layer in self.lmodel.lmodels:
                if layer.type == "relu":
                    self.indexToReluvar.append([m.addVar(vtype=GRB.BINARY, name='reluVar') for _ in range(layer.size)])
                    layer.setReluVar(self.indexToReluvar[-1])
                else:
                    continue
            m.update()'''

    def addGurobiConstrsBetweenLayers(self, m: 'gp.Model'):
        # 处理输入层到输出层的约束
        preLayer = self.lmodel.inputLayer
        for lidx, layer in enumerate(self.lmodel.lmodels):
            # constrMethod=-1 表示使用全局的约束方法，为以后的优化做准备
            # constrMethod= 0 表示使用精确地混合整型编码方式进行约束
            # constrMethod= 1 表示使用三角松弛进行约束
            constrMethod = 0
            layer.addConstr(preLayer, m, constrMethod=constrMethod)
            preLayer = layer

    def addManualConstraints(self, m: gp.Model):
        m.update()
        # 理论上在这里添加输出层上的约束
        constraints = self.spec.outputConstr
        if isinstance(constraints, Disjunctive):
            for constr in constraints.constraints:
                if constr[0] == "VarVar":
                    var1 = m.getVarByName("Y{}".format(constr[1]))
                    relation = constr[2]
                    var2 = m.getVarByName("Y{}".format(constr[3]))
                    if relation == "GT":
                        m.addConstr(var1 <= var2)
                    elif relation == "LT":
                        m.addConstr(var1 >= var2)
                    elif relation == "EQ":
                        pass
                    else:
                        raise Exception("输出约束关系异常")
                elif constr[0] == "VarValue":
                    var = m.getVarByName("Y{}".format(constr[1]))
                    relation = constr[2]
                    value = m.getVarByName("Y{}".format(constr[3]))
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
            additionalConstrBinVar = [
                m.addVar(vtype=GRB.BINARY) for _ in range(constrlength)]
            for (i, constr) in enumerate(constraints.constraints):
                if constr[0] == "VarVar":
                    var1 = m.getVarByName(constr[1])
                    relation = constr[2]
                    var2 = m.getVarByName(constr[3])
                    if relation == "GT":
                        m.addConstr(
                            (additionalConstrBinVar[i] == 1) >> (var1 <= var2))
                    elif relation == "LT":
                        m.addConstr(
                            (additionalConstrBinVar[i] == 1) >> (var1 >= var2))
                    elif relation == "EQ":
                        pass
                    else:
                        raise Exception("输出约束关系异常")
                elif constr[0] == "VarValue":
                    var = m.getVarByName(constr[1])
                    relation = constr[2]
                    value = int(constr[3])
                    if relation == "GT":
                        m.addConstr(
                            (additionalConstrBinVar[i] == 1) >> (var <= value))
                    elif relation == "LT":
                        m.addConstr(
                            (additionalConstrBinVar[i] == 1) >> (var >= value))
                    elif relation == "EQ":
                        m.addConstr(var >= value)
                    else:
                        raise Exception("输出约束关系异常")
                m.update()
            m.addConstr(quicksum(additionalConstrBinVar) >= 1)
            m.addConstr(quicksum(additionalConstrBinVar) <= constrlength)
            m.update()

    def optimize_bounds(self):
        m = gp.Model("approx")
        m.Params.OutputFlag = 0
        opt_bounds_num = 0
        # 添加输入层的COUTINUE变量，并添加输入层约束
        self.indexToVar.append(
            [
                m.addVar(
                    ub=self.lmodel.inputLayer.var_bounds_out["ub"][i],
                    lb=self.lmodel.inputLayer.var_bounds_out["lb"][i],
                    vtype=GRB.CONTINUOUS,
                    name='X{}'.format(str(i))
                ) for i in range(self.lmodel.inputLayer.size)
            ]
        )
        self.lmodel.inputLayer.setVar(self.indexToVar[-1])
        m.update()

        self.indexToVar.append(
            [
                m.addVar(
                    ub=self.lmodel.lmodels[0].var_bounds_out["ub"][i],
                    lb=self.lmodel.lmodels[0].var_bounds_out["lb"][i],
                    vtype=GRB.CONTINUOUS,
                ) for i in range(self.lmodel.lmodels[0].size)
            ]
        )
        self.lmodel.lmodels[0].setVar(self.indexToVar[-1])
        m.update()

        # 添加lmodel[1]及其之后的变量，因为对于lmodel[0]，即使是区间算术，也很难造成较大的误差因此，不进行边界优化

        for i in range(1, self.lmodel.layerNum - 2):
            w = self.lmodel.lmodels[i].weight
            x = self.lmodel.lmodels[i - 1].var
            b = self.lmodel.lmodels[i].bias
            wx_plus_b = np.dot(w, x) + b
            for j in range(self.lmodel.lmodels[i].size):
                ub = self.lmodel.lmodels[i].var_bounds_in["ub"][j]
                lb = self.lmodel.lmodels[i].var_bounds_in["lb"][j]

                obj = m.addVar(
                    ub=ub,
                    lb=lb
                )

                '''
                TODO
                这里好像不对吧，如果是用三角松弛来优化边界，这里为什么就一条constr，一个三角松弛不应该改是三条约束么？
                '''
                m.addConstr(obj == wx_plus_b[j])

                m.update()

                m.setObjective(obj, GRB.MINIMIZE)
                m.optimize()
                if m.status == GRB.Status.OPTIMAL:
                    lb = obj.X
                    if lb > self.lmodel.lmodels[i].var_bounds_in["lb"][j]:
                        opt_bounds_num += 1
                        self.lmodel.lmodels[i].var_bounds_in["lb"][j] = lb

                m.update()

                m.setObjective(obj, GRB.MAXIMIZE)
                m.optimize()
                if m.status == GRB.Status.OPTIMAL:
                    ub = obj.X
                    if ub < self.lmodel.lmodels[i].var_bounds_in["ub"][j]:
                        opt_bounds_num += 1
                        self.lmodel.lmodels[i].var_bounds_in["ub"][j] = ub

                m.remove(m.getVars()[-1])
                m.remove(m.getConstrs()[-1])
                m.update()
            self.indexToVar.append(
                [
                    m.addVar(
                        ub=self.lmodel.lmodels[i].var_bounds_out["ub"][k],
                        lb=self.lmodel.lmodels[i].var_bounds_out["lb"][k],
                        vtype=GRB.CONTINUOUS,
                    ) for k in range(self.lmodel.lmodels[i].size)
                ]
            )
            self.lmodel.lmodels[i].setVar(self.indexToVar[-1])
            m.update()

        return opt_bounds_num