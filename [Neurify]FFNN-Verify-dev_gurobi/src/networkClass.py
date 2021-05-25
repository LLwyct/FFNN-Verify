import keras, sys
import numpy as np
import pickle
from Layer import Layer, InputLayer, ReluLayer, LinearLayer
from typing import List, Union, Optional
from numpy import ndarray
from keras.models import load_model
from property import getNormaliseInput
from LinearFunctions import LinearFunctions
from options import GlobalSetting

class Network:
    def __init__(self, path="", fmtType="h5", propertyReadyToVerify=-1, imgPklFilePath="", verifyType="acas"):
        self.networkFilePath: str = path
        self.netFmtType: str = fmtType
        self.verifyType = verifyType
        self.propertyIndexReadyToVerify: int = propertyReadyToVerify
        self.layerNum: int = 0
        self.eachLayerNums: List[int] = []
        self.inputLmodel: Optional[InputLayer] = None
        self.lmodel: List[Union[ReluLayer, LinearLayer]] = []
        '''
        weight矩阵，长度为layerNum - 1
        当计算第i层第j个节点的值时，使用向量weight[i] * x_{i-1}
        '''
        self.weights: List[ndarray] = []
        self.biases: List[ndarray] = []

        if self.verifyType == "mnist":
            if imgPklFilePath == "":
                raise Exception("未提供mnist类型验证所需图片文件路径")
            self.radius = 0.05
            with open(imgPklFilePath, "rb") as pickle_file:
                data = pickle.load(pickle_file)
            self.image = data[0]
            self.label = data[1]
        self.init()

    def init(self):
        if self.netFmtType == "nnet" and self.networkFilePath != "":
            self.readFromNnet()
        elif self.netFmtType == "h5" and self.networkFilePath != "":
            self.readFromH5()
        else:
            raise IOError
        # 初始化/预计算每一层，包括输入层，隐藏层，输出层的边界
        self.initBounds()

    def initBounds(self):
        if self.verifyType == 'acas':
            # 初始化acasxu类型输入层的边界
            res = getNormaliseInput(self.propertyIndexReadyToVerify)
            self.inputLmodel.setBounds(res[0], res[1])
        else:
            # 初始化mnist类型输入层的边界
            input_lower_bounds = np.maximum(self.image - self.radius, 0)
            input_upper_bounds = np.minimum(self.image + self.radius, 1)
            for i in range(len(input_upper_bounds)):
                if input_upper_bounds[i] < input_lower_bounds[i]:
                    raise Exception("Value Error in input bounds", i)
            self.inputLmodel.setBounds(input_lower_bounds, input_upper_bounds)

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
            res = getNormaliseInput(self.propertyIndexReadyToVerify)
            self.inputLmodel.var_bounds_in_cmp["ub"] = self.inputLmodel.var_bounds_out_cmp["ub"] = res[0]
            self.inputLmodel.var_bounds_in_cmp["lb"] = self.inputLmodel.var_bounds_out_cmp["lb"] = res[1]
            self.symbolIntervalPropation_sia_and_slr()

    def intervalPropation(self):
        preLayer_u = self.inputLmodel.var_bounds_out["ub"]
        preLayer_l = self.inputLmodel.var_bounds_out["lb"]
        for layer in self.lmodel:
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
                preLayer_u = layer.var_bounds_out["ub"] = np.maximum(layer.var_bounds_in["ub"], np.zeros(u_hat.shape))
                preLayer_l = layer.var_bounds_out["lb"] = np.maximum(layer.var_bounds_in["lb"], np.zeros(l_hat.shape))
            elif isinstance(layer, LinearLayer):
                # 如果当前层是linear层，则不需要经过ReLU激活函数
                preLayer_u = layer.var_bounds_out["ub"] = layer.var_bounds_in["ub"]
                preLayer_l = layer.var_bounds_out["lb"] = layer.var_bounds_in["lb"]
            for i in range(len(layer.var_bounds_in["ub"])):
                if layer.var_bounds_in["ub"][i] < layer.var_bounds_in["lb"][i]:
                    raise Exception

    def symbolIntervalPropation_0sia_or_1slr(self, method):
        inputLayerSize = self.inputLmodel.size
        self.inputLmodel.bound_equations["out"]["lb"] = LinearFunctions(np.identity(inputLayerSize), np.zeros(inputLayerSize))
        self.inputLmodel.bound_equations["out"]["ub"] = LinearFunctions(np.identity(inputLayerSize), np.zeros(inputLayerSize))

        preLayer = self.inputLmodel
        for layer in self.lmodel:
            if layer.type == "relu":
                assert isinstance(layer, ReluLayer)
                layer.compute_Eq_and_bounds_0sia_or_1slr(preLayer, self.inputLmodel, method)
            elif layer.type == "linear":
                assert isinstance(layer, LinearLayer)
                layer.compute_Eq_and_bounds(preLayer, self.inputLmodel)
            preLayer = layer

    def symbolIntervalPropation_sia_and_slr(self):
        inputLayerSize = self.inputLmodel.size
        self.inputLmodel.bound_equations["out"]["lb"] = LinearFunctions(np.identity(inputLayerSize),
                                                                        np.zeros(inputLayerSize))
        self.inputLmodel.bound_equations["out"]["ub"] = LinearFunctions(np.identity(inputLayerSize),
                                                                        np.zeros(inputLayerSize))

        self.inputLmodel.bound_equations_cmp["out"]["lb"] = LinearFunctions(np.identity(inputLayerSize),
                                                                        np.zeros(inputLayerSize))
        self.inputLmodel.bound_equations_cmp["out"]["ub"] = LinearFunctions(np.identity(inputLayerSize),
                                                                         np.zeros(inputLayerSize))
        preLayer = self.inputLmodel
        for layer in self.lmodel:
            if layer.type == "relu":
                assert isinstance(layer, ReluLayer)
                layer.compute_Eq_and_bounds_sia_and_slr(preLayer, self.inputLmodel)
            elif layer.type == "linear":
                assert isinstance(layer, LinearLayer)
                layer.compute_Eq_and_bounds(preLayer, self.inputLmodel)
            preLayer = layer

        for layer in self.lmodel:
            slr_out_diff = 0
            merge_out_diff = 0
            maxUpper_out_sia = -1 * sys.maxsize
            minLower_out_sia = +1 * sys.maxsize
            maxUpper_out_slr = -1 * sys.maxsize
            minLower_out_slr = +1 * sys.maxsize
            for i in range(layer.size):
                slr_out_diff += layer.var_bounds_out["ub"][i] - layer.var_bounds_out["lb"][i]
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
            if layer.id < self.layerNum - 1:
                layer.var_bounds_in["ub"] = np.minimum(layer.var_bounds_in["ub"], layer.var_bounds_in_cmp["ub"])
                layer.var_bounds_in["lb"] = np.maximum(layer.var_bounds_in["lb"], layer.var_bounds_in_cmp["lb"])
                #layer.var_bounds_out["ub"] = np.minimum(layer.var_bounds_out["ub"], layer.var_bounds_out_cmp["ub"])
                #layer.var_bounds_out["lb"] = np.maximum(layer.var_bounds_out["lb"], layer.var_bounds_out_cmp["lb"])

            for i in range(layer.size):
                merge_out_diff += layer.var_bounds_out["ub"][i] - layer.var_bounds_out["lb"][i]
            print(slr_out_diff, merge_out_diff)

    def readFromNnet(self):
        if self.networkFilePath == "":
            raise IOError
        with open(self.networkFilePath, "r") as f:
            # 读取网络层数
            line = f.readline()
            if line == "":
                raise Exception("IOError")

            self.layerNum = int(line)

            # 读取每一层的节点数
            line = f.readline()
            if line == "":
                raise Exception("IOError")
            self.eachLayerNums = [int(num) for num in line.split(" ")]

            # 创建每一层的weight矩阵，暂时为空
            for layer in range(self.layerNum - 1):
                # 如果第0层是5个节点，第1层是50个节点，那么weights[0]应该是一个50*5的矩阵，注意这里是反过来的
                self.weights.append(np.empty([self.eachLayerNums[layer+1], self.eachLayerNums[layer]], float))

            # 创建每一层的bias矩阵，暂时为空
            for layer in range(self.layerNum - 1):
                self.biases.append(np.empty([self.eachLayerNums[layer + 1]], float))

            for layer in range(self.layerNum - 1):
                # 读取权重
                weightMatrixRowNum = self.weights[layer].shape[0]
                weight = []
                for row in range(weightMatrixRowNum):
                    matrixLine = f.readline()
                    if matrixLine == "":
                        raise Exception("IOError")
                    weight.append([float(num) for num in matrixLine.split(" ") if num != "\n"])
                self.weights[layer] = np.array(weight, dtype=float)

                # 读取bias
                biasMatrixRowNum = self.weights[layer].shape[0]
                bias = []
                for row in range(biasMatrixRowNum):
                    biasLine = f.readline()
                    if biasLine == "":
                        raise Exception("IOError")
                    bias.append(float(biasLine))
                self.biases[layer] = np.array(bias, dtype=float)

    def readFromH5(self):
        net_model = load_model(self.networkFilePath, compile=False)
        self.layerNum = len(net_model.layers) + 1
        self.eachLayerNums.append(net_model.layers[0].get_weights()[0].shape[0])
        self.inputLmodel = InputLayer(0, size=self.eachLayerNums[0])
        for i, layer in enumerate(net_model.layers):
            if layer.activation == keras.activations.relu:
                self.lmodel.append(ReluLayer(
                    i+1,
                    layer.get_weights()[0].T,
                    layer.get_weights()[1],
                    layer_type="relu",
                ))
            elif layer.activation == keras.activations.linear:
                self.lmodel.append(LinearLayer(
                    i+1,
                    layer.get_weights()[0].T,
                    layer.get_weights()[1],
                    layer_type="linear"
                ))
            else:
                raise IOError
            self.weights.append(layer.get_weights()[0].T)
            self.biases.append(layer.get_weights()[1])
            self.eachLayerNums.append(len(layer.get_weights()[1]))
        pass