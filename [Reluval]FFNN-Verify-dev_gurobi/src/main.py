import os
import argparse
from options import Options
from solverClass import Solver
from networkClass import Network


def getOptions():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--npath",
        required=True,
        help="the neural net workpath for h5 file"
    )
    parse.add_argument(
        "--type",
        choices=["rea", "rob"],
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
    solver = Solver(network, propertyFilePath)
    '''
    手动管理输出约束 for property 3，不应该输出COC，即Y[0]
    在acas中，输出越小，期望越高
    因此应该编写的反例即为COC期望最高（值最小）
    y0 < y1
    y0 < y2
    y0 < y3
    y0 < y4
    '''
    solver.m.addConstr(network.lmodel[-1].var[0] <= network.lmodel[-1].var[1])
    solver.m.addConstr(network.lmodel[-1].var[0] <= network.lmodel[-1].var[2])
    solver.m.addConstr(network.lmodel[-1].var[0] <= network.lmodel[-1].var[3])
    solver.m.addConstr(network.lmodel[-1].var[0] <= network.lmodel[-1].var[4])
    solver.m.update()
    # end
    '''
    gurobi已经提供了关于容忍误差，所以此处不需要考虑舍入问题
    '''
    solver.solve()
    print(networkFilePath)

def mainForRun():
    networkFileName = "acas_1_6.h5"
    propertyFileName = "property_3.txt"
    networkFilePath = os.path.abspath(os.path.join("../resources/Acas", networkFileName))
    propertyFilePath = os.path.abspath(os.path.join("../resources", propertyFileName))
    network = Network(networkFilePath, fmtType="h5", propertyReadyToVerify=3)
    solver = Solver(network, propertyFilePath)
    # 手动管理输出约束，输入约束在property.py中
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

if __name__ == "__main__":
    # 默认作为脚本使用，如为了方便测试可以使用mainForRun
    mainForRun()
    # mainForOuterScript()



