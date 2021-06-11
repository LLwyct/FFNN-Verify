import copy
import numpy as np
import sys
from options import GlobalSetting
from Layer import ReluLayer, LinearLayer
from LinearFunction import LinearFunction
from LayerModel import LayerModel

class VerifyModel:
    def __init__(self, lmodel, spec):
        self.lmodel: LayerModel = copy.deepcopy(lmodel)
        self.spec = copy.deepcopy(spec)
        self.verifyType = "acas"
        self.lmodel.loadSpec(self.spec)
    
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
            pass
            # res = getNormaliseInput(self.propertyIndexReadyToVerify)
            # self.lmodel.inputLayer.var_bounds_in_cmp["ub"] = self.lmodel.inputLayer.var_bounds_out_cmp["ub"] = res[0]
            # self.lmodel.inputLayer.var_bounds_in_cmp["lb"] = self.lmodel.inputLayer.var_bounds_out_cmp["lb"] = res[1]
            # self.symbolIntervalPropation_sia_and_slr()

    def intervalPropation(self):
        preLayer_u = self.lmodel.lmodels.inputLayer.var_bounds_out["ub"]
        preLayer_l = self.lmodel.lmodels.inputLayer.var_bounds_out["lb"]
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
        for layer in self.lmodel:
            if layer.type == "relu":
                assert isinstance(layer, ReluLayer)
                layer.compute_Eq_and_bounds_sia_and_slr(
                    preLayer, self.lmodel.inputLayer)
            elif layer.type == "linear":
                assert isinstance(layer, LinearLayer)
                layer.compute_Eq_and_bounds(preLayer, self.lmodel.inputLayer)
            preLayer = layer

        for layer in self.lmodel:
            slr_out_diff = 0
            # merge_out_diff = 0
            maxUpper_out_sia = -1 * sys.maxsize
            minLower_out_sia = +1 * sys.maxsize
            maxUpper_out_slr = -1 * sys.maxsize
            minLower_out_slr = +1 * sys.maxsize
            for i in range(layer.size):
                slr_out_diff += layer.var_bounds_out["ub"][i] - \
                    layer.var_bounds_out["lb"][i]
                if layer.var_bounds_out_cmp["ub"][i] > maxUpper_out_sia:
                    maxUpper_out_sia = layer.var_bounds_out_cmp["ub"][i]
                if layer.var_bounds_out_cmp["lb"][i] < minLower_out_sia:
                    minLower_out_sia = layer.var_bounds_out_cmp["lb"][i]
                if layer.var_bounds_out["ub"][i] > maxUpper_out_slr:
                    maxUpper_out_slr = layer.var_bounds_out["ub"][i]
                if layer.var_bounds_out["lb"][i] < minLower_out_slr:
                    minLower_out_slr = layer.var_bounds_out["lb"][i]
            print(layer.id, "outter")
            print(maxUpper_out_sia, maxUpper_out_slr)
            if layer.id < self.lmodel.layerNum - 1:
                layer.var_bounds_in["ub"] = np.minimum(
                    layer.var_bounds_in["ub"], layer.var_bounds_in_cmp["ub"])
                layer.var_bounds_in["lb"] = np.maximum(
                    layer.var_bounds_in["lb"], layer.var_bounds_in_cmp["lb"])
                #layer.var_bounds_out["ub"] = np.minimum(layer.var_bounds_out["ub"], layer.var_bounds_out_cmp["ub"])
                #layer.var_bounds_out["lb"] = np.maximum(layer.var_bounds_out["lb"], layer.var_bounds_out_cmp["lb"])

            '''for i in range(layer.size):
                merge_out_diff += layer.var_bounds_out["ub"][i] - layer.var_bounds_out["lb"][i]
            print(slr_out_diff, merge_out_diff)'''

    def getFixedNodeRatio(self) -> float:
        return self.lmodel.getFixedNodeRatio()
