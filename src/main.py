import os, sys

from networkClass import Network
from solverClass import Solver
from gurobipy import GRB


if __name__ == "__main__":
    networkFileName = "acas_1_1.h5"
    propertyFileName = "property_3.txt"
    networkFilePath = os.path.abspath(os.path.join("../resources/Acas", networkFileName))
    propertyFilePath = os.path.abspath(os.path.join("../resources", propertyFileName))
    network = Network(networkFilePath, type="h5")
    solver = Solver(network, propertyFilePath)
    network.intervalPropagate()
    # 手动管理输出约束
    solver.m.addConstr(solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][1])
    solver.m.addConstr(solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][2])
    solver.m.addConstr(solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][3])
    solver.m.addConstr(solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][4])
    '''
    gurobi已经提供了关于容忍误差，所以此处不需要考虑舍入问题
    '''
    solver.m.setObjective(0, GRB.MAXIMIZE)
    solver.solve()
