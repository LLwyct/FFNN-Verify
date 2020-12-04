from mip import Model, BINARY, INTEGER, xsum
from reader import loadNetwork, loadProperty
from network import Network
from sys import stdout

M = 100


def addNetworkConstraints(m: Model, net: Network, varIndex = [], reluIndex= []):
    # 添加网络节点变量
    for layer in range(net.layerNum):
        varIndex.append([m.add_var() for i in range(net.eachLayerNums[layer])])

    # 添加relu节点变量
    for layer in range(net.layerNum):
        if layer == 0:
            reluIndex.append([])
        else:
            reluIndex.append([m.add_var(var_type=BINARY) for i in range(net.eachLayerNums[layer])])

    # 对于每一个节点添加不含ReLU节点的线性约束，公式1和公式2
    for currentLayerIndex, currentLayer in enumerate(varIndex):
        # 处理当前层currentLayerIndex的全部节点
        lastLayerIndex = currentLayerIndex - 1
        lastLayerNodeNum = net.eachLayerNums[lastLayerIndex]
        if currentLayerIndex == 0:
            continue
        else:
            for nodeIndex, node in enumerate(currentLayer):
                # node is a var
                tempIter = []
                for i in range(lastLayerNodeNum):
                    tempIter.append(varIndex[lastLayerIndex][i] * net.weights[lastLayerIndex][i][nodeIndex])
                # 公式1
                m += xsum(i for i in tempIter) <= node
                # 公式3
                m += node >= 0

    # 对于每一个节点添加ReLU线性约束，公式2和公式4
    for currentLayerIndex, currentLayer in enumerate(varIndex):
        # 处理当前层currentLayerIndex的全部节点
        lastLayerIndex = currentLayerIndex - 1
        lastLayerNodeNum = net.eachLayerNums[lastLayerIndex]
        if currentLayerIndex == 0:
            continue
        else:
            for nodeIndex, node in enumerate(currentLayer):
                # 公式4
                m += (node <= M * reluIndex[currentLayerIndex][nodeIndex])
                # 公式2
                tempIter = []
                for i in range(lastLayerNodeNum):
                    tempIter.append(varIndex[lastLayerIndex][i] * net.weights[lastLayerIndex][i][nodeIndex])
                m += ((xsum(i for i in tempIter) + M * (1 - reluIndex[currentLayerIndex][nodeIndex])) >= node)
    return varIndex, reluIndex


def addManualConstraints(m: Model, net: Network, varIndex, reluIndex, path=""):
    inputConstraints, outputConstraints = loadProperty(path)
    lastLayerIndex = net.layerNum - 1
    for inputConstraint in inputConstraints:
        varIdx        = inputConstraint[0]
        equationType    = inputConstraint[1]
        scalar          = inputConstraint[2]
        if equationType == 0:
            m += varIndex[0][varIdx] == scalar
        elif equationType == 1:
            m += varIndex[0][varIdx] <= scalar
        elif equationType == 2:
            m += varIndex[0][varIdx] >= scalar
    for outputConstraint in outputConstraints:
        varIdx          = outputConstraint[0]
        equationType    = outputConstraint[1]
        scalar          = outputConstraint[2]
        if equationType == 0:
            m += varIndex[lastLayerIndex][varIdx] == scalar
        elif equationType == 1:
            m += varIndex[lastLayerIndex][varIdx] <= scalar
        elif equationType == 2:
            m += varIndex[lastLayerIndex][varIdx] >= scalar

if __name__ == '__main__':
    m = Model()
    generalVar, reluVar = [], []

    network = loadNetwork("./input.txt")
    addNetworkConstraints(m, network, generalVar, reluVar)
    addManualConstraints(m, network, generalVar, reluVar, "./property.txt")

    m.optimize()

    if m.num_solutions:
        stdout.write("-----------------solutation found!-----------------\n")
        for i in range(network.layerNum):
            for j, node in enumerate(generalVar[i]):
                print("x_{}{}:{}".format(i, j, node.x))

        for i in range(network.layerNum):
            for j, node in enumerate(reluVar[i]):
                print("&_{}{}:{}".format(i, j, node.x))
        stdout.write("-----------------solutation found!-----------------\n")
    else:
        print("-----------------solutation not found!-----------------\n")
        print("unsat")
        print("-----------------solutation not found!-----------------\n")
