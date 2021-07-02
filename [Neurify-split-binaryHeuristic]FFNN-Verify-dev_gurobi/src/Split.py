from numpy import ndarray

class Split:
    def __init__(self, id: str, upper: ndarray, lower: ndarray) -> None:
        self.id: str = id
        self.up: ndarray = upper
        self.lo: ndarray = lower
    
    @staticmethod
    def createByInterval(id: str, spec):
        return Split(id, spec.inputBounds["ub"], spec.inputBounds["lb"])