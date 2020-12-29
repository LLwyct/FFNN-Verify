import numpy as np
from keras.models import load_model
import keras
from Layer import Layer
from property import getNormaliseInput


class Network:
    def __init__(self, path="", type="h5", propertyReadyToVerify=0):
        # 网络文件路径
        self.networkFilePath: str = path
        self.type = type
        self.propertyIndexReadyToVerify = propertyReadyToVerify
        # 网络层数，包括输入输出层
        self.layerNum: int = -1
        # 每一层的节点数
        self.eachLayerNums: list = []
        self.inputLmodel = Layer()
        self.lmodel: list = []
        '''
        weight矩阵，长度为layerNum - 1
        当计算第i层第j个节点的值时，使用向量x_i-1 * weight[i-1][]
        '''
        self.weights: list = []

        # bias矩阵，长度也为layerNum - 1
        self.biases:list = []

        self.bounds = []

        self.init()


    def init(self):
        if self.type == "nnet" and self.networkFilePath != "":
            self.read()
        elif self.type == "h5" and self.networkFilePath != "":
            self.readFromH5()
        else:
            raise IOError
        self.initBounds()
        self.intervalCompute()


    def initBounds(self):
        res = getNormaliseInput(self.propertyIndexReadyToVerify)
        self.inputLmodel.var_bounds_out["lb"] = res[0]
        self.inputLmodel.var_bounds_out["ub"] = res[1]
        pass

    def intervalCompute(self):
        preLayer_u = self.inputLmodel.var_bounds_out["ub"]
        preLayer_l = self.inputLmodel.var_bounds_out["lb"]
        for layer in self.lmodel:
            assert isinstance(layer, Layer)
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
            # 到这里于venus甚至运行的数据都一模一样
            if layer.type == "relu":
                preLayer_u = layer.var_bounds_out["ub"] = np.maximum(layer.var_bounds_in["ub"], np.zeros(u_hat.shape))
                preLayer_l = layer.var_bounds_out["lb"] = np.maximum(layer.var_bounds_in["lb"], np.zeros(l_hat.shape))
            elif layer.type == "linear":
                preLayer_u = layer.var_bounds_out["ub"] = layer.var_bounds_in["ub"]
                preLayer_l = layer.var_bounds_out["lb"] = layer.var_bounds_in["lb"]

    def read(self):
        self.weights = []
        self.biases = []
        if self.networkFilePath == "":
            return
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
        self.inputLmodel.size = self.eachLayerNums[0]
        for layer in net_model.layers:
            if layer.activation == keras.activations.relu:
                self.lmodel.append(Layer(
                    "relu",
                    layer.get_weights()[0].T,
                    layer.get_weights()[1]
                ))
            elif layer.activation == keras.activations.linear:
                self.lmodel.append(Layer(
                    "linear",
                    layer.get_weights()[0].T,
                    layer.get_weights()[1]
                ))
            else:
                self.lmodel.append(Layer(
                    "unknown"
                ))
            self.weights.append(layer.get_weights()[0].T)
            self.biases.append(layer.get_weights()[1])
            self.eachLayerNums.append(len(layer.get_weights()[1]))
        pass


class Node:
    def __init__(self):
        self.lb = None
        self.ub = None