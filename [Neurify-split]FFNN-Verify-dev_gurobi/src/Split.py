from typing import Tuple
from numpy import ndarray

from Specification import Specification

class Split:
    def __init__(self, id: int, upper: ndarray, lower: ndarray) -> None:
        self.id: int = id
        self.up: ndarray = upper
        self.lo: ndarray = lower
    
    @staticmethod
    def createByInterval(id: int, spec: Specification):
        return Split(id, spec.inputBounds["ub"], spec.inputBounds["lb"])