import os
import argparse
from timeit import default_timer as timer

from numpy.lib.function_base import average
from Network import Network
from MainVerify import MainVerify
from options import GlobalSetting

def main_for_run(verify_type, case):

    if verify_type == "acas":
        start = timer()
        networkFileName = "acas_1_{}.h5".format(case)
        networkFilePath = os.path.abspath(os.path.join("../resources/Acas", networkFileName))
        network: 'Network' = Network(networkFilePath, fmtType="h5", propertyReadyToVerify=3, verifyType="acas")
        spec = network.getInitialSpec()
        mainVerifier = MainVerify(network, spec)
        isSat, splittingTime, solverTime, max_jobs_num =  mainVerifier.verify()
        end = timer()
        with open("acas.log", "at") as f:
            f.write("{}\n".format(networkFileName))
            f.write("finally result : {}\n".format("SAT" if isSat else "UNSAT"))
            f.write("total time     : {:.2f}\n".format(end - start))
            f.write("splitting time : {:.2f}\n".format(splittingTime))
            f.write("solver time    : {:.2f} {}\n ".format(solverTime, max_jobs_num))
            if max_jobs_num != 0:
                f.write("avg solver time: {:.2f}\n\n".format(solverTime / max_jobs_num))
        print("finally result: ", "SAT" if isSat else "UNSAT")
        print("splitting time: ", splittingTime)
        print("solver time   : ", solverTime)
        print("total time    : ", end - start)

if __name__ == '__main__':
    with open("acas.log", "at") as f:
        f.write("splitting process num: {}\n".format(GlobalSetting.splitting_processes_num))
        f.write("vmodel solver process num: {}\n".format(GlobalSetting.vmodel_verify_processes_num))
        f.write("splitting fixed ratio theshold: {}\n".format(GlobalSetting.SPLIT_THRESHOLD))
        f.write("presolver method: {}\n".format(GlobalSetting.preSolveMethod))
        f.write("use bounds optimised?: {}\n".format(GlobalSetting.use_bounds_opt))
        f.write("------------------------------------\n")
    for i in range(1, 10):
        main_for_run("acas", i)
