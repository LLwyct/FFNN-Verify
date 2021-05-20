import os
import argparse
from options import Options
from solverClass import Solver
from networkClass import Network
from gurobipy import GRB, quicksum
import pickle

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
    solver.solve(options.checkType)
    print(networkFilePath)

def mainForRun(case, verifyType="acas"):
    if verifyType == "acas":
        networkFileName = "acas_1_{}.h5".format(case)
        networkFilePath = os.path.abspath(os.path.join("../resources/Acas", networkFileName))
        network = Network(networkFilePath, fmtType="h5", propertyReadyToVerify=3, verifyType="acas")
        solver = Solver(network)
        solver.solve(verifyType)
        print(networkFileName)
    elif verifyType == "mnist":
        imgPklFileName = "im{}.pkl".format(case)
        networkFileName = "mnist-net.h5"
        imgPklFilePath = os.path.abspath(os.path.join("../resources/Mnist/evaluation_images", imgPklFileName))
        networkFilePath = os.path.abspath(os.path.join("../resources/Mnist", networkFileName))
        network = Network(networkFilePath, fmtType="h5", imgPklFilePath=imgPklFilePath, verifyType="mnist")
        solver = Solver(network)
        # 手动管理输出约束，输入约束在property.py中
        oC = [solver.m.addVar(vtype=GRB.BINARY) for i in range(10)]
        solver.m.update()
        for i in range(10):
            if i == network.label:
                solver.m.remove(oC[i])
                continue
            else:
                solver.m.addConstr((oC[i] == 1) >> (network.lmodel[-1].var[i] >= network.lmodel[-1].var[network.label]))
        solver.m.update()
        del oC[network.label]
        solver.m.addConstr(quicksum(oC) <= 9)
        solver.m.addConstr(quicksum(oC) >= 1)
        solver.m.update()
        solver.solve(type)
        print(imgPklFileName)

    '''
    gurobi已经提供了关于容忍误差，所以此处不需要考虑舍入问题
    '''
    # solver.m.setObjective()


if __name__ == "__main__":
    # 默认作为脚本使用，如为了方便测试可以使用mainForRun
    '''
    type = ["mnist", "acas"]
    mnist 用于测试图片鲁棒性类的网络
    acas  用于测试属性安全类的网络
    '''
    for i in range(1, 2):
        mainForRun(i, verifyType="acas")
    # mainForOuterScript()