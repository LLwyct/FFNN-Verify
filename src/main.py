from networkClass import Network
from solverClass import Solver
from mip import minimize, xsum


if __name__ == "__main__":
    network = Network("../example/ACASXU_experimental_v2a_1_3.nnet")
    solver = Solver(network, "../example/property_3.txt")
    solver.m += solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][1]
    solver.m += solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][2]
    solver.m += solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][3]
    solver.m += solver.indexToVar[network.layerNum - 1][0] <= solver.indexToVar[network.layerNum - 1][4]
    epslions = []
    for i in range(network.layerNum):
        if i == 0:
            continue
        for j in range(network.eachLayerNums[i]):
            epslions.append(solver.indexToEpsilon[i][j])
            solver.m += solver.indexToEpsilon[i][j] >= 0
    solver.m += (xsum(epslion for epslion in epslions) <= 1)
    solver.m.objective = minimize(xsum(epslion for epslion in epslions))
    solver.solve()
    print(solver.m.objective_value)