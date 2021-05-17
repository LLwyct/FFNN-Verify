from mip import Model, BINARY, INTEGER, xsum
from networkClass import Network

M = 1000000

class Solver:
    def __init__(self, network: Network, propertyFile: str):
        self.m = Model()
        self.net = network
        self.indexToVar = []
        self.indexToReluvar = []
        self.indexToEpsilon = []
        self.propertyFile = propertyFile
        self.initEpsilon()
        # 初始化网络级约束
        self.addNetworkConstraints(self.m, self.net, self.indexToVar, self.indexToReluvar, self.indexToEpsilon)
        # 初始化人工约束
        self.addManualConstraints()

    def initEpsilon(self):
        for i in range(self.net.layerNum):
            if i == 0:
                self.indexToEpsilon.append([])
            else:
                self.indexToEpsilon.append([self.m.add_var() for i in range(self.net.eachLayerNums[i])])

    @staticmethod
    def addNetworkConstraints(m: Model, net: Network, indexToVar: list, indexToReluvar: list, indexToEpsilon):
        # 添加网络节点变量
        for layer in range(net.layerNum):
            # 添加实数变量
            indexToVar.append([m.add_var()
                               for i in range(net.eachLayerNums[layer])])

        # 添加relu节点变量，是二进制变量，0代表非激活y=0，1代表激活y=x
        for layer in range(net.layerNum):
            # 第一层input不添加relu节点
            if layer == 0:
                indexToReluvar.append([])
            else:
                # 输出层默认加relu激活函数
                indexToReluvar.append([m.add_var(var_type=BINARY)
                                       for i in range(net.eachLayerNums[layer])])

        # 对于每一个节点添加网络级线性约束
        for curLayerIdx, curLayer in enumerate(indexToVar):
            if curLayerIdx == 0:
                continue
            # 处理当前层curLayer的全部节点
            lastLayerIdx = curLayerIdx - 1
            lastLayerNodeNum = net.eachLayerNums[lastLayerIdx]
            if lastLayerNodeNum != len(indexToVar[lastLayerIdx]):
                raise
            # 先取出来bias，防止频繁访问net对象
            bias = net.biases[lastLayerIdx]
            weight = net.weights[lastLayerIdx]
            for curNodeIdx, curNode in enumerate(curLayer):
                # node is a var
                # 公式1
                tempIter = []
                for idx, varnodeInLastLayer in enumerate(indexToVar[lastLayerIdx]):
                    tempIter.append(
                        varnodeInLastLayer *
                        weight[curNodeIdx][idx])
                m += (xsum(i for i in tempIter) +
                        bias[curNodeIdx] - indexToEpsilon[curLayerIdx][curNodeIdx]) <= curNode

                # 公式2
                tempIter = []
                for idx, varnodeInLastLayer in enumerate(indexToVar[lastLayerIdx]):
                    tempIter.append(
                        varnodeInLastLayer *
                        weight[curNodeIdx][idx])
                m += (xsum(i for i in tempIter) + bias[curNodeIdx]) + M * (
                    1 - indexToReluvar[curLayerIdx][curNodeIdx]) + indexToEpsilon[curLayerIdx][curNodeIdx] >= curNode

                # 公式3
                m += curNode >= 0

                # 公式4
                m += (curNode <= M *
                    indexToReluvar[curLayerIdx][curNodeIdx])

    def addManualConstraints(self):
        # 添加输入层输出层上的约束
        inputConstraints, outputConstraints = self.loadProperty()
        finalLayerIndex = self.net.layerNum - 1
        for inputConstraint in inputConstraints:
            varIdx          = inputConstraint[0]
            equationType    = inputConstraint[1]
            scalar          = inputConstraint[2]
            if equationType == 0:
                self.m += self.indexToVar[0][varIdx] == scalar
            elif equationType == 1:
                self.m += self.indexToVar[0][varIdx] <= scalar
            elif equationType == 2:
                self.m += self.indexToVar[0][varIdx] >= scalar
        for outputConstraint in outputConstraints:
            varIdx          = outputConstraint[0]
            equationType    = outputConstraint[1]
            scalar          = outputConstraint[2]
            if equationType == 0:
                self.m += self.indexToVar[finalLayerIndex][varIdx] == scalar
            elif equationType == 1:
                self.m += self.indexToVar[finalLayerIndex][varIdx] <= scalar
            elif equationType == 2:
                self.m += self.indexToVar[finalLayerIndex][varIdx] >= scalar

    def loadProperty(self):
        inputConstraints = []
        outputConstraints = []
        with open(self.propertyFile, "r") as f:
            line = f.readline()
            while line:
                if line == "":
                    raise Exception("IOError")
                var_equation_scala = line.split(" ")
                varName_varIndex = var_equation_scala[0].split("_")
                equation = var_equation_scala[1]
                scalar = var_equation_scala[2]
                varName = varName_varIndex[0]
                varIndex = varName_varIndex[1]
                if varName == "x" and equation == "==":
                    inputConstraints.append([int(varIndex), 0, float(scalar)])
                elif varName == "x" and equation == "<=":
                    inputConstraints.append([int(varIndex), 1, float(scalar)])
                elif varName == "x" and equation == ">=":
                    inputConstraints.append([int(varIndex), 2, float(scalar)])
                elif varName == "y" and equation == "==":
                    outputConstraints.append([int(varIndex), 0, float(scalar)])
                elif varName == "y" and equation == "<=":
                    outputConstraints.append([int(varIndex), 1, float(scalar)])
                elif varName == "y" and equation == ">=":
                    outputConstraints.append([int(varIndex), 2, float(scalar)])
                else:
                    raise Exception("IOError")
                line = f.readline()
        return inputConstraints, outputConstraints

    def solve(self):
        self.m.optimize()
        if self.m.num_solutions:
            print("-----------------solutation found!-----------------")
            for i in range(self.net.layerNum):
                for j, node in enumerate(self.indexToVar[i]):
                    print("x_{}{}:{}".format(i, j, node.x))

            for i in range(self.net.layerNum):
                for j, node in enumerate(self.indexToReluvar[i]):
                    print("&_{}{}:{}".format(i, j, node.x))
            print("-----------------solutation found!-----------------")
        else:
            print("-----------------solutation not found!-----------------")
            print("unsat")
            print("-----------------solutation not found!-----------------")
