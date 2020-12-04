import numpy as np


class Network:
    def __init__(self, path):
        self.layerNum = 0
        self.path = path
        self.weights = None
        self.eachLayerNums = None
        self.biases = None
        self.read(path)

    def read(self, path):
        self.weights = []
        self.biases = []
        with open(self.path, "r") as f:
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
                self.weights.append(np.empty([self.eachLayerNums[layer], self.eachLayerNums[layer + 1]], float))

            # 读取权重
            for layer in range(self.layerNum - 1):
                weightMatrixRowNum = self.weights[layer].shape[0]
                weight = []
                for row in range(weightMatrixRowNum):
                    matrixLine = f.readline()
                    if matrixLine == "":
                        raise Exception("IOError")
                    weight.append([float(num) for num in matrixLine.split(" ")])
                self.weights[layer] = np.array(weight, dtype=float)

            # 创建每一层的bias矩阵，暂时为空
            for layer in range(self.layerNum - 1):
                self.biases.append(np.empty([self.eachLayerNums[layer + 1]], float))

            # 读取bias
            for layer in range(self.layerNum - 1):
                biasMatrixRowNum = self.weights[layer].shape[1]
                bias = []
                for row in range(biasMatrixRowNum):
                    biasLine = f.readline()
                    if biasLine == "":
                        raise Exception("IOError")
                    bias.append([float(num) for num in biasLine.split(" ")])
                self.biases[layer] = np.array(bias, dtype=float)

    def log(self):
        print(
            self.layerNum + "\n"
            + self.eachLayerNums
        )
