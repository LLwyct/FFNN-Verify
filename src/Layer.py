from numpy import ndarray

class L:
    def __init__(self):
        pass


class Layer:
    def __init__(self, layer_type="unknown", w = None, b = None):
        # "linear" | "relu" | "input" | "unknown"
        self.type: str = layer_type
        self.weight: ndarray = w
        self.bias: ndarray = b
        self.size: int = -1
        if b is None and type == "input":
            pass
        elif b is not None:
            self.size = b.size
        self.var_bounds_in = {
            "ub": None,
            "lb": None
        }
        self.var_bounds_out = {
            "ub": None,
            "lb": None
        }
        self.var = None
        self.reluVar = None
        self.nodeList = []

    def setVar(self, varlist):
        if self.var is None:
            self.var = varlist
        else:
            pass

    def setReluVar(self, reluVarList):
        if self.type == "relu" and self.reluVar is None:
            self.reluVar = reluVarList

    def getVar(self):
        if self.var is not None:
            return self.var
        else:
            pass

    def getReluVar(self):
        if self.type == "relu" and self.reluVar is not None:
            return self.reluVar
        else:
            pass

