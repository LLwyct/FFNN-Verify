import copy
from typing import List, Set
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
from property import acas_denormalise_output, acas_properties
from keras.models import load_model

class VerifyModel:
    def __init__(self, id: str, lmodel: 'LayerModel', spec: 'Specification'):
        self.id: str = id
        self.lmodel: 'LayerModel' = copy.deepcopy(lmodel)
        self.spec: 'Specification' = copy.deepcopy(spec)
        self.verifyType = "acas"
        self.lmodel.loadSpec(self.spec) # load input bounds
        self.gmodel = None
        self.indexToVar: List[List] = []
        self.indexToReluvar: List[List] = []
        self.networkModel = None

    def verify(self, networkModel) -> bool:
        '''
        networkModel 一般是指神经网络的H5模型
        当sat满足约束的取反时返回False
        当unsat不满足约束的取反时返回True
        '''
        self.networkModel = load_model(networkModel, compile=False)
        self.gmodel = gp.Model()
        self.gmodel.Params.OutputFlag = 0
        self.indexToVar = []
        self.indexToReluvar: List[List] = []
        while True and GlobalSetting.use_bounds_opt is True:
            # 属于预处理的步骤，使用三角近似来优化边界，要在initAllBounds之后执行
            self.indexToVar = []
            opt_num = self.optimize_bounds()
            #print("optmise bounds number: ", opt_num)
            if opt_num == 0:
                break
            break
        if GlobalSetting.use_binary_heuristic_method == 0:
            # 先添加Gurobi Model的全部变量，CONTINUES变量和BINARY变量
            self.addGurobiVarFromLayerModel(self.gmodel)
            # 添加层与层之间变量的约束
            self.addGurobiConstrsBetweenLayers(self.gmodel)
            # 添加输出层人为规定的安全约束
            self.addManualConstraints(self.gmodel)
            self.gmodel.optimize()
            if self.gmodel.status == GRB.OPTIMAL:
                # print("launch slover", self.id, False)
                return False
            else:
                # print("launch slover", self.id, True)
                return True
        elif GlobalSetting.use_binary_heuristic_method == 1:
            self.gmodel.Params.outputFlag = 0
            res = self.checkSATWithbinaryHeuristicMethod(self.gmodel)
            return res


    def checkSATWithbinaryHeuristicMethod(self, m) -> bool:
        notFixedNodesNum: int = 0
        for layer in self.lmodel.lmodels:
            if layer.type == "relu":
                notFixedNodesNum += layer.getNotFixedNodeNum()
        lo: int = 0
        hi: int = notFixedNodesNum // 2
        if self.spec.verifyType == "mnist" and notFixedNodesNum > 30:
            hi = 30
        #print("restNum", notFixedNodesNum)
        hi = 40
        while lo <= hi:
            # init 到初始状态
            self.gmodel.reset()
            self.gmodel.remove(self.gmodel.getConstrs())
            self.gmodel.remove(self.gmodel.getGenConstrs())
            self.gmodel.remove(self.gmodel.getVars())
            self.indexToVar = []
            self.indexToReluvar: List[List] = []
            self.gmodel.update()
            self.addGurobiVarFromLayerModel(self.gmodel)
            self.addGurobiConstrsBetweenLayers(self.gmodel, hi)
            self.addManualConstraints(self.gmodel)
            self.gmodel.optimize()
            #print("restHi", hi)
            if self.gmodel.status == GRB.OPTIMAL:
                inputVar = [m.getVarByName("X{}".format(i)) for i in range(self.lmodel.inputLayer.size)]
                inputVar = [Var.X for Var in inputVar]
                res = self.isTrulyConterExample(inputVar)
                if res:
                    return False
                if hi == 0:
                    return False
                hi = hi // 2
                if self.spec.verifyType == "mnist":
                    # 如果是mnist数据集，初始的notFixedNodesNum会比较大，并且结果随剩余未固定节点减少变化不明显，因此要加速
                    hi = int(hi / 1.5)
            else:
                # print("binary search end, sat")
                # 没有解，说明是找不到反例因此是sat的，在使用了松弛后依然是sat则原本必定是sat的
                return True


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

        
        # 添加relu节点变量，是二进制变量，0代表非激活y=0，1代表激活y=x
        for layer in self.lmodel.lmodels:
            if layer.type == "relu":
                self.indexToReluvar.append([m.addVar(vtype=GRB.BINARY, name='reluVar') for _ in range(layer.size)])
                layer.setReluVar(self.indexToReluvar[-1])
            else:
                continue
        m.update()

    def addGurobiConstrsBetweenLayers(self, m: 'gp.Model', restNum: int = 0):
        # 处理输入层到输出层的约束
        preLayer = self.lmodel.inputLayer
        if restNum <= 2:
            for lidx, layer in enumerate(self.lmodel.lmodels):
                # constrMethod=-1 表示使用全局的约束方法，为以后的优化做准备
                # constrMethod= 0 表示使用精确地混合整型编码方式进行约束
                # constrMethod= 1 表示使用三角松弛进行约束
                constrMethod = 0
                layer.addConstr(preLayer, m, constrMethod=constrMethod)
                preLayer = layer
        elif restNum > 0:
            restNum = restNum
            for lidx, layer in enumerate(self.lmodel.lmodels):
                constrMethod = 0
                restNum = layer.addConstr_BinaryHeuristic(
                    preLayer,
                    m,
                    constrMethod=constrMethod,
                    utrnl=None,
                    restNum=restNum
                )
                preLayer = layer

    def addManualConstraints(self, m: gp.Model):
        if self.spec.verifyType == "mnist":
            label = self.spec.label
            oC = [m.addVar(vtype=GRB.BINARY, name='additional') for i in range(10)]
            m.update()
            for i in range(10):
                if i == label:
                    m.remove(oC[i])
                    continue
                else:
                    m.addConstr(
                        (oC[i] == 1) >> (self.lmodel.lmodels[-1].var[i] >= self.lmodel.lmodels[-1].var[label])
                    )
            m.update()
            del oC[label]
            m.addConstr(quicksum(oC) <= 9)
            m.addConstr(quicksum(oC) >= 1)
            m.update()
            return
        elif self.spec.verifyType == "acas":
            m.update()
            # 理论上在这里添加输出层上的约束
            constraints = acas_properties[self.spec.propertyReadyToVerify]["outputConstraints"][-1]
            if isinstance(constraints, Disjunctive):
                for constr in constraints.constraints:
                    # 这里后续需要优化，constr[0]是什么？ 不利于阅读，应该改成字典类型constr['type']
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
                    ub=self.lmodel.inputLayer.var_bounds_out["ub"][i],
                    lb=self.lmodel.inputLayer.var_bounds_out["lb"][i],
                    vtype=GRB.CONTINUOUS,
                    name='X{}'.format(str(i))
                ) for i in range(self.lmodel.inputLayer.size)
            ]
        )
        self.lmodel.inputLayer.setVar(self.indexToVar[-1])
        m.update()

        # 添加lmodel[0]的变量，因为对于第一个隐藏层来说它的边界由输入层经过一次区间传播而来，一定是精确的
        # 因此不需要对其边界进行优化
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

        self.lmodel.lmodels[0].addConstr(self.lmodel.inputLayer, m, 1)

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
                # 这里只需要创建一个==的约束，因为这里优化的是激活前的上下界
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
            # 当这层的每一个节点都分别优化完了， 更新了in的区间范围，别忘了更新out的区间范围
            self.lmodel.lmodels[i].var_bounds_out["ub"] = np.maximum(self.lmodel.lmodels[i].var_bounds_in["ub"], 0)
            self.lmodel.lmodels[i].var_bounds_out["lb"] = np.maximum(self.lmodel.lmodels[i].var_bounds_in["lb"], 0)

            # 当这层的每一个节点都分别优化完了，把这一层的全部节点添加到模型， 并建立与上一层的约束，进行下一轮循环
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
            self.lmodel.lmodels[i].addConstr(self.lmodel.lmodels[i-1], m, 1)
            m.update()

        return opt_bounds_num

    def isTrulyConterExample(self, inputVar) -> bool:
        from keras.models import load_model
        ans: bool = False
        outputVar = self.networkModel(np.array([inputVar]))[0]
        if self.spec.verifyType == "acas":
            realOutputVar = acas_denormalise_output(np.array(outputVar))
            constraints = acas_properties[self.spec.propertyReadyToVerify]["outputConstraints"][-1]
            if isinstance(constraints, Disjunctive):
                # 这里是或约束，且是原始的正例或约束，因此只要一条子约束满足，则不是反例
                for constr in constraints.constraints:
                    # 这里后续需要优化，constr[0]是什么？ 不利于阅读，应该改成字典类型constr[‘type’]
                    if constr[0] == "VarVar":
                        var1Idx = int(constr[1][1:])
                        relation = constr[2]
                        var2Idx = int(constr[3][1:])
                        if relation == "GT":
                            if realOutputVar[var1Idx] - realOutputVar[var2Idx] > 0.000001:
                                ans = True
                                break
                        elif relation == "LT":
                            if realOutputVar[var1Idx] - realOutputVar[var2Idx] < -0.000001:
                                ans = True
                                break
                        elif relation == "EQ":
                            pass
                        else:
                            raise Exception("输出约束关系异常")
                    elif constr[0] == "VarValue":
                        varIdx = constr[1][1:]
                        relation = constr[2]
                        value = float(constr[3])
                        if relation == "GT":
                            if realOutputVar[varIdx] - value >= 0.000001:
                                ans = True
                        elif relation == "LT":
                            if realOutputVar[varIdx] - value <= -0.000001:
                                ans = True
                        elif relation == "EQ":
                            pass
                        else:
                            raise Exception("输出约束关系异常")
            elif isinstance(constraints, Conjunctive):
                pass
        elif self.spec.verifyType == "mnist":
            # 在mnist数据集里，这里是且约束，只要有一条反子约束不满足则是真实反例
            label = self.spec.label
            for i in range(self.lmodel.lmodels[-1].size):
                if i != label and outputVar[i] - outputVar[label] >= -0.000001:
                    ans = False
                    break
                else:
                    ans = True
        return not ans