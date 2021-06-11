from Layer import InputLayer, Layer
from Layer import ReluLayer
from Layer import LinearLayer
import keras
from keras.models import load_model
from typing import Optional
from Specification import Specification

class LayerModel:

    def __init__(self):
        self.layerNum = -1
        self.inputLayer: Optional[InputLayer] = None
        self.eachLayerNums = []
        self.lmodels: Layer = []

    # 使用h5或nnet文件来初始化network的每一层weights和bias参数
    def initLayerModel(self, networkFilePath, netFmtType):
        if netFmtType == "h5":
            self.loadFromH5(networkFilePath)
        elif netFmtType == "nnet":
            self.loadFromNnet(networkFilePath)

    # 使用spec初始化输入层的上下界
    def loadSpec(self, spec:Specification):
        self.inputLayer.setBounds(spec.inputBounds["ub"], spec.inputBounds["lb"])

    def getFixedNodeRatio(self):
        totalNodeNum = 0
        fixedNodeNum = 0
        for layer in self.lmodels:
            if layer.type == "relu":
                totalNodeNum += layer.size
                fixedNodeNum += layer.getFixedNodeNum()
        return fixedNodeNum / totalNodeNum

    def loadFromH5(self, networkFilePath):
        net_model = load_model(networkFilePath, compile=False)
        self.layerNum = len(net_model.layers) + 1
        self.eachLayerNums.append(net_model.layers[0].get_weights()[0].shape[0])
        self.inputLayer = InputLayer(0, size=self.eachLayerNums[0])
        for i, layer in enumerate(net_model.layers):
            if layer.activation == keras.activations.relu:
                self.lmodels.append(ReluLayer(
                    i + 1,
                    layer.get_weights()[0].T,
                    layer.get_weights()[1],
                    layer_type="relu",
                ))
            elif layer.activation == keras.activations.linear:
                self.lmodels.append(LinearLayer(
                    i + 1,
                    layer.get_weights()[0].T,
                    layer.get_weights()[1],
                    layer_type="linear"
                ))
            else:
                raise IOError
            self.eachLayerNums.append(len(layer.get_weights()[1]))

    def loadFromNnet(self, networkFilePath):
        pass
        '''
        
        :param networkFilePath: 
        :return: 
        
        if networkFilePath == "":
            raise IOError
        with open(networkFilePath, "r") as f:
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
                self.weights.append(np.empty([self.eachLayerNums[layer + 1], self.eachLayerNums[layer]], float))

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
        '''
