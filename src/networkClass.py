import numpy as np


class Network:
    def __init__(self, path):
        # 网络文件路径
        self.networkFilePath: str = path

        # 网络层数，包括输入输出层
        self.layerNum: int = 0

        # 每一层的节点数
        self.eachLayerNums: list = None

        '''
        weight矩阵，长度为layerNum - 1
        当计算第i层第j个节点的值时，使用向量x_i-1 * weight[i-1][]
        '''
        self.weights: list = None

        # bias矩阵，长度也为layerNum - 1
        self.biases = None

        self.read()

    def read(self):
        self.weights = []
        self.biases = []
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
