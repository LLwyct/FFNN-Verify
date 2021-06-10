import os
import argparse
#from options import Options
#from solverClass import Solver
from networkClass import Network
from gurobipy import GRB, quicksum
#from options import GlobalSetting
from MainVerify import MainVerify


def main_for_run(verify_type, case):
    if verify_type == "acas":
        networkFileName = "acas_1_{}.h5".format(case)
        networkFilePath = os.path.abspath(os.path.join("../resources/Acas", networkFileName))
        network: Network = Network(networkFilePath, fmtType="h5", propertyReadyToVerify=3, verifyType="acas")
        spec = network.getInitialSpec()
        mainVerifier = MainVerify(network, spec)

if __name__ == '__main__':
    main_for_run("acas", 1)
