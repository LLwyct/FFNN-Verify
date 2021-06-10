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
