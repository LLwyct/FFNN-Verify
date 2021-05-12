import keras
import numpy as np
import pickle
from Layer import Layer, InputLayer, ReluLayer, LinearLayer
from typing import List, Union, Optional
from numpy import ndarray
from keras.models import load_model
from property import getNormaliseInput
from LinearFunctions import LinearFunctions

class Network:
    def __init__(self, path="", fmtType="h5", propertyReadyToVerify=-1, imgPklFilePath=""):
        self.networkFilePath: str = path
        self.netFmtType: str = fmtType
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

        if imgPklFilePath != "":
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
        self.initBounds()
        # 在mipverify中需要区间传播，所以不能注释掉这句话，Layer.py中使用带有上下界的编码
        # self.intervalPropation()
        # 在这里实现朴素符号传播，即ReLUval中的符号传播逻辑，并非venus中的符号传播
        self.symbolIntervalPropation()
        pass


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

    def initBounds(self):
        if self.propertyIndexReadyToVerify != -1:
            res = getNormaliseInput(self.propertyIndexReadyToVerify)
            self.inputLmodel.setBounds(res[0], res[1])
        else:
            input_lower_bounds = np.maximum(self.image - self.radius, 0)
            input_upper_bounds = np.minimum(self.image + self.radius, 1)
            for i in range(len(input_upper_bounds)):
                if input_upper_bounds[i] < input_lower_bounds[i]:
                    print("Value Error in input bounds", i)
            self.inputLmodel.setBounds(input_lower_bounds, input_upper_bounds)
            pass

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

    def symbolIntervalPropation(self):
        inputLayerSize = self.inputLmodel.size
        self.inputLmodel.bound_equations["out"]["lb"] = LinearFunctions(np.identity(inputLayerSize), np.zeros(inputLayerSize))
        self.inputLmodel.bound_equations["out"]["ub"] = LinearFunctions(np.identity(inputLayerSize), np.zeros(inputLayerSize))

        preLayer = self.inputLmodel
        for layer in self.lmodel:
            layer.compute_Eq_and_bounds_sia(preLayer, self.inputLmodel)
            preLayer = layer

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

