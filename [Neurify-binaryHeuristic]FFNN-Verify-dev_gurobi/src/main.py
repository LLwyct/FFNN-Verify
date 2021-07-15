import os
import argparse
from options import Options
from solverClass import Solver
from networkClass import Network
from gurobipy import GRB, quicksum
from options import GlobalSetting
from timeit import default_timer as timer
def getOptions():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--npath",
        required=True,
        help="the neural net workpath for h5 file"
    )
    parse.add_argument(
        "--type",
        choices=["acas", "mnist"],
        default="rea",
        help="which type of property will be verified"
    )
    parse.add_argument(
        "--prop",
        required=True,
        type=int,
        help="which property will be verified"
    )
    ARGS = parse.parse_args()
    proOptions = Options(ARGS.npath, ARGS.type, ARGS.prop)
    return proOptions

def mainForOuterScript():
    options = getOptions()
    networkFilePath = os.path.abspath(os.path.join("../resources", options.netPath))
    propertyFilePath = os.path.abspath(os.path.join("../resources"))
    network = Network(networkFilePath, fmtType="h5", propertyReadyToVerify=options.property)
    solver = Solver(network)
    # end
    '''
    gurobi已经提供了关于容忍误差，所以此处不需要考虑舍入问题
    '''
    solver.solve()
    print(networkFilePath)

def mainForRun(case, verifyType="acas"):
    if verifyType == "acas":
        start = timer()
        networkFileName = "acas_2_{}.h5".format(case)
        networkFilePath = os.path.abspath(os.path.join("../resources/Acas", networkFileName))
        network = Network(networkFilePath, fmtType="h5", propertyReadyToVerify=3, verifyType="acas")
        solver = Solver(network)
        res = solver.verify()
        end = timer()
        if GlobalSetting.write_to_file:
            with open("result.log", "at") as f:
                f.write("{} {} {} {:.2f}\n".format(networkFileName, case, res, end - start))
        print(">{} {} {} {:.2f}\n".format(networkFileName, case, res, end - start))
    elif verifyType == "mnist":
        start = timer()
        if GlobalSetting.preSolveMethod == 4:
            GlobalSetting.preSolveMethod = 3
        imgPklFileName = "im{}.pkl".format(case)
        networkFileName = "mnist-net.h5"
        imgPklFilePath = os.path.abspath(os.path.join("../resources/Mnist/evaluation_images", imgPklFileName))
        networkFilePath = os.path.abspath(os.path.join("../resources/Mnist", networkFileName))
        network = Network(networkFilePath, fmtType="h5", imgPklFilePath=imgPklFilePath, verifyType="mnist")
        solver = Solver(network)
        res = solver.verify()
        end = timer()
        if GlobalSetting.write_to_file:
            with open("result.log", "at") as f:
                f.write("{} {} {} {:.2f}\n".format(imgPklFileName, case, res, end - start))
        print(">{} {} {} {:.2f}\n".format(imgPklFileName, case, res, end - start))
    '''
    gurobi已经提供了关于容忍误差，所以此处不需要考虑舍入问题
    '''

if __name__ == "__main__":
    # 默认作为脚本使用，如为了方便测试可以使用mainForRun
    '''
    type = ["mnist", "acas"]
    mnist 用于测试图片鲁棒性类的网络
    acas  用于测试属性安全类的网络
    '''
    for i in range(1, 10):
        mainForRun(i, verifyType="acas")
    # mainForOuterScript()