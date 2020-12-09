from networkClass import Network
from solverClass import Solver

if __name__ == "__main__":
    network = Network("../input.txt")
    solver = Solver(network, "../property.txt")
    solver.solve()