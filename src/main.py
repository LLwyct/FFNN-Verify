from networkClass import Network
from solverClass import Solver
from mip import minimize, xsum
from gurobipy import quicksum, GRB
import gurobipy as gp

if __name__ == "__main__":
    h5path = "C:/Users/liwenchi/Documents/GitHub/FFNN-Verify/resources/Acas/acas_1_1.h5"
    network = Network(h5path, h5=True)
    solver = Solver(network, "../resources/property_3.txt")
    # solver.m += solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][1]
    # solver.m += solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][2]
    # solver.m += solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][3]
    # solver.m += solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][4]
    solver.m.addConstr(solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][1])
    solver.m.addConstr(solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][2])
    solver.m.addConstr(solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][3])
    solver.m.addConstr(solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][4])

    epslions = []
    for i in range(network.layerNum):
        if i == 0:
            continue
        for j in range(network.eachLayerNums[i]):
            epslions.append(solver.indexToEpsilon[i][j])
    solver.m.setObjective(quicksum(epslion for epslion in epslions), GRB.MAXIMIZE)
    solver.solve()
    # print(solver.m.objVal)