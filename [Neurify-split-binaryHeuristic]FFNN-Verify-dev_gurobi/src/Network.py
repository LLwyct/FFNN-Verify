import pickle
from Layer import InputLayer, ReluLayer, LinearLayer
from typing import List, Union, Optional
from numpy import ndarray
from LayerModel import LayerModel
from Specification import Specification


class Network:
    def __init__(self, path="", fmtType="h5", propertyReadyToVerify=-1, imgPklFilePath="", verifyType="acas"):
        self.networkFilePath: str = path
        self.netFmtType: str = fmtType
        self.verifyType = verifyType
        self.networkModel = None
        self.propertyIndexReadyToVerify: int = propertyReadyToVerify
        self.inputLmodel: Optional['InputLayer'] = None
        self.lmodel = LayerModel()
        self.spec = Specification(verifyType=verifyType, propertyReadyToVerify=propertyReadyToVerify)
        self.image = None
        self.label = None
        '''
        weight矩阵，长度为layerNum - 1
        当计算第i层第j个节点的值时，使用向量weight[i] * x_{i-1}
        '''
        self.weights: List['ndarray'] = []
        self.biases: List['ndarray'] = []

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
        # 如果self.verifyType是mnist，则image，label为None
        self.spec.load(self.propertyIndexReadyToVerify, self.verifyType, self.image, self.label)
        # net_model 用于预测反例
        self.networkModel = self.lmodel.initLayerModel(self.networkFilePath, self.netFmtType)

    def getInitialSpec(self):
        return self.spec

