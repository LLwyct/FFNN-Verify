from numpy.core.fromnumeric import reshape
from Split import Split
from property import getNormaliseInput, acas_properties
import copy

class Specification:
    def __init__(self, ub=None, lb=None):
        self.inputBounds = {
            "ub": None,
            "lb": None
        }
        self.outputBounds = {
            "ub": None,
            "lb": None
        }
        self.outputConstr = None

    def load(self, propIndex, type):
        if type == "acas":
            inputBounds = getNormaliseInput(propIndex)
            self.inputBounds["lb"] = inputBounds[0]
            self.inputBounds["ub"] = inputBounds[1]
            self.outputConstr = acas_properties[propIndex]["outputConstraints"][-1]
        elif type == "mnist":
            pass

    def setInputBounds(self, ub, lb):
        self.inputBounds = {
            "ub": ub,
            "lb": lb
        }

    def getInputBounds(self):
        return (self.inputBounds["ub"], self.inputBounds["lb"])

    def clone(self):
        return copy.deepcopy(self)

    def resetFromSplit(self, split: Split):
        newSpec = copy.deepcopy(self)
        newSpec.setInputBounds(split.up, split.lo)
        return newSpec
