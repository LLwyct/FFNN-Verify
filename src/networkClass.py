import numpy as np
from keras.models import load_model


class Network:
    def __init__(self, path="", type="h5"):
        # 网络文件路径
        self.networkFilePath: str = path
        self.type = type
        # 网络层数，包括输入输出层
        self.layerNum: int = -1

        # 每一层的节点数
        self.eachLayerNums: list = []

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

    def initBounds(self):
        for i in range(self.layerNum):
            self.bounds.append([])
            for j in range(self.eachLayerNums[i]):
                self.bounds[i].append(Node())
        pass

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
        for i in range(len(net_model.layers)):
            self.weights.append(net_model.layers[i].get_weights()[0].T)
            self.biases.append(net_model.layers[i].get_weights()[1])
            self.eachLayerNums.append(len(net_model.layers[i].get_weights()[1]))

    def intervalPropagate(self):
        for i in range(1, self.layerNum):
            pass
        pass


class Node:
    def __init__(self):
        self.lb = None
        self.ub = None