from networkClass import Network
from solverClass import Solver
from mip import minimize, xsum


if __name__ == "__main__":
    # mat = loadmat("../resources/Acas/ACASXU_run2a_1_1_batch_2000.mat")
    h5path = "C:/Users/liwenchi/Documents/GitHub/FFNN-Verify/resources/Acas/acas_1_2.h5"
    network = Network(h5path, h5=True)
    solver = Solver(network, "../resources/property_3.txt")
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
    solver.m.objective = minimize(xsum(epslion for epslion in epslions))
    solver.solve()
    print(solver.m.objective_value)