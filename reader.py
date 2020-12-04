import numpy as np
from network import Network
import fileinput

def loadNetwork(path: str = "") -> Network:
    try:
        network = Network(path)
    except IOError:
        print("Error: 没有找到文件或读取文件失败")
    return network


def loadProperty(path: str = ""):
    inputConstraints = []
    outputConstraints = []
    with open(path, "r") as f:
        line = f.readline()
        while line:
            if line == "":
                raise Exception("IOError")
            var_equation_scala = line.split(" ")
            varName_varIndex    = var_equation_scala[0].split("_")
            equation            = var_equation_scala[1]
            scalar              = var_equation_scala[2]
            varName             = varName_varIndex[0]
            varIndex            = varName_varIndex[1]
            if varName == "x" and equation == "==":
                inputConstraints.append([int(varIndex), 0, float(scalar)])
            elif varName == "x" and equation == "<=":
                inputConstraints.append([int(varIndex), 1, float(scalar)])
            elif varName == "x" and equation == ">=":
                inputConstraints.append([int(varIndex), 2, float(scalar)])
            elif varName == "y" and equation == "==":
                outputConstraints.append([int(varIndex), 0, float(scalar)])
            elif varName == "y" and equation == "<=":
                outputConstraints.append([int(varIndex), 1, float(scalar)])
            elif varName == "y" and equation == ">=":
                outputConstraints.append([int(varIndex), 2, float(scalar)])
            else:
                raise Exception("IOError")
            line = f.readline()
    return inputConstraints, outputConstraints



