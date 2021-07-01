from typing import Dict, Optional, Tuple
from numpy import ndarray
from ConstraintFormula import Formulation
from Split import Split
from property import getNormaliseInput, acas_properties
import copy

class Specification:
    def __init__(self, ub: Optional[ndarray] = None, lb: Optional[ndarray] = None):
        self.inputBounds: Dict[str, Optional[ndarray]] = {
            "ub": None,
            "lb": None
        }
        self.outputBounds: Dict[str, Optional[ndarray]] = {
            "ub": None,
            "lb": None
        }
        self.outputConstr: Optional[Formulation] = None

    def load(self, propIndex: int, type: str):
        if type == "acas":
            inputBounds = getNormaliseInput(propIndex)
            self.inputBounds["lb"] = inputBounds[0]
            self.inputBounds["ub"] = inputBounds[1]
            self.outputConstr = acas_properties[propIndex]["outputConstraints"][-1]
        elif type == "mnist":
            pass

    def setInputBounds(self, ub:ndarray, lb:ndarray):
        '''
            利用ub lb去更新该spec的上下界
        '''
        self.inputBounds = {
            "ub": ub,
            "lb": lb
        }

    def getInputBounds(self) -> Tuple[Optional[ndarray], Optional[ndarray]]:
        '''
            返回该Spec的输入上下界的元组 (upper, lower)，upper lower可能为None
        '''
        return (self.inputBounds["ub"], self.inputBounds["lb"])

    def clone(self):
        '''
            返回一个自己的deepcopy
        '''
        return copy.deepcopy(self)

    def resetFromSplit(self, split: Split):
        '''
            使用Split的区间去更新Spec，而不改变output约束
        '''
        newSpec = copy.deepcopy(self)
        newSpec.setInputBounds(split.up, split.lo)
        return newSpec
