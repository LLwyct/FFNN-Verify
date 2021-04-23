import keras
import numpy as np
from Layer import Layer, InputLayer, ReluLayer, LinearLayer
from typing import List, Union, Optional
from numpy import ndarray
from keras.models import load_model
from property import getNormaliseInput


class Network:
    def __init__(self, path="", fmtType="h5", propertyReadyToVerify=0):
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

        self.init()

    def init(self):
        if self.netFmtType == "nnet" and self.networkFilePath != "":
            self.readFromNnet()
        elif self.netFmtType == "h5" and self.networkFilePath != "":
            self.readFromH5()
        else:
            raise IOError
        self.initBounds()
        # 在nsverify中不需要区间传播，所以注释掉这句话，Layer.py中使用bigM编码
        # self.intervalCompute()
        pass


    def readFromH5(self):
        net_model = load_model(self.networkFilePath, compile=False)
        self.layerNum = len(net_model.layers) + 1
        self.eachLayerNums.append(net_model.layers[0].get_weights()[0].shape[0])
        self.inputLmodel = InputLayer(size=self.eachLayerNums[0])
        for layer in net_model.layers:
            if layer.activation == keras.activations.relu:
                self.lmodel.append(ReluLayer(
                    layer.get_weights()[0].T,
                    layer.get_weights()[1],
                    layer_type="relu",
                ))
            elif layer.activation == keras.activations.linear:
                self.lmodel.append(LinearLayer(
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
        res = getNormaliseInput(self.propertyIndexReadyToVerify)
        self.inputLmodel.setBounds(res[0], res[1])
        pass

    def intervalCompute(self):
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
                preLayer_u = layer.var_bounds_out["ub"] = np.maximum(layer.var_bounds_in["ub"], np.zeros(u_hat.shape))
                preLayer_l = layer.var_bounds_out["lb"] = np.maximum(layer.var_bounds_in["lb"], np.zeros(l_hat.shape))
            elif isinstance(layer, LinearLayer):
                preLayer_u = layer.var_bounds_out["ub"] = layer.var_bounds_in["ub"]
                preLayer_l = layer.var_bounds_out["lb"] = layer.var_bounds_in["lb"]

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


