import os
from gurobipy import GRB
from networkClass import Network
from solverClass import Solver


if __name__ == "__main__":
    networkFileName = "acas_1_6.h5"
    propertyFileName = "property_3.txt"
    networkFilePath = os.path.abspath(os.path.join("../resources/Acas", networkFileName))
    propertyFilePath = os.path.abspath(os.path.join("../resources", propertyFileName))
    network = Network(networkFilePath, type="h5", propertyReadyToVerify=3)
    solver = Solver(network, propertyFilePath)
    # 手动管理输出约束
    solver.m.addConstr(network.lmodel[-1].var[0] <= network.lmodel[-1].var[1])
    solver.m.addConstr(network.lmodel[-1].var[0] <= network.lmodel[-1].var[2])
    solver.m.addConstr(network.lmodel[-1].var[0] <= network.lmodel[-1].var[3])
    solver.m.addConstr(network.lmodel[-1].var[0] <= network.lmodel[-1].var[4])
    solver.m.update()
    '''
    gurobi已经提供了关于容忍误差，所以此处不需要考虑舍入问题
    '''
    # solver.m.setObjective()
    solver.solve()
    print(networkFileName)
