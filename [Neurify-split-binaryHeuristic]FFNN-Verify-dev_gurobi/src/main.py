import os
import argparse
from timeit import default_timer as timer
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
        isSat, splittingTime, solverTime, max_jobs_num, truthTime =  mainVerifier.verify()
        end = timer()
        if GlobalSetting.write_to_file:
            with open("result.log", "at") as f:
                f.write("{} {} {} {:.2f}\n\n".format(networkFileName, case, isSat, truthTime))
        print(">{} {} {} {:.2f}\n\n".format(networkFileName, case, isSat, truthTime))
        return truthTime
    if verify_type == "mnist":
        start = timer()
        imgPklFileName = "im{}.pkl".format(case)
        networkFileName = "mnist-net.h5"
        imgPklFilePath = os.path.abspath(os.path.join("../resources/Mnist/evaluation_images", imgPklFileName))
        networkFilePath = os.path.abspath(os.path.join("../resources/Mnist", networkFileName))
        network = Network(networkFilePath, fmtType="h5", imgPklFilePath=imgPklFilePath, verifyType="mnist")
        spec = network.getInitialSpec()
        mainVerifier = MainVerify(network, spec)
        isSat, splittingTime, solverTime, max_jobs_num, truthTime =  mainVerifier.verify()
        end = timer()
        if GlobalSetting.write_to_file:
            with open("result.log", "at") as f:
                f.write("{} {} {} {:.2f}".format(imgPklFileName, case, isSat, truthTime))
        print(">{} {} {} {:.2f}\n\n".format(imgPklFileName, case, isSat, truthTime))

if __name__ == '__main__':
    if GlobalSetting.write_to_file:
        with open("result.log", "at") as f:
            f.write("splitting process num: {}\n".format(GlobalSetting.splitting_processes_num))
            f.write("vmodel solver process num: {}\n".format(GlobalSetting.vmodel_verify_processes_num))
            f.write("splitting fixed ratio theshold: {}\n".format(GlobalSetting.SPLIT_THRESHOLD))
            f.write("presolver method: {}\n".format(GlobalSetting.preSolveMethod))
            f.write("use bounds optimised?: {}\n".format(GlobalSetting.use_bounds_opt))
            f.write("use_binary_heuristic_method?: {}\n".format(GlobalSetting.use_binary_heuristic_method))
            f.write("------------------------------------\n")
    times = []
    for i in range(1, 10):
        t = main_for_run("acas", i)
        times.append(t)
    if GlobalSetting.write_to_file:
        with open("result.log", "at") as f:
            f.write("average time: {:.2f}\n\n".format(sum(times) / len(times)))