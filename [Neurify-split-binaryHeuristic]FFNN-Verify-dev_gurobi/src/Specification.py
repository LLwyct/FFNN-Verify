from pickle import NONE
from typing import Dict, Optional, Tuple
from numpy import ndarray
from ConstraintFormula import Formulation
from Split import Split
from options import GlobalSetting
from property import getNormaliseInput, acas_properties
import numpy as np
import copy

class Specification:
    def __init__(self, ub: Optional[ndarray] = None, lb: Optional[ndarray] = None, verifyType = "acas", propertyReadyToVerify = -1):
        self.verifyType = verifyType
        self.propertyReadyToVerify = propertyReadyToVerify
        self.inputBounds: Dict[str, Optional[ndarray]] = {
            "ub": None,
            "lb": None
        }
        self.outputBounds: Dict[str, Optional[ndarray]] = {
            "ub": None,
            "lb": None
        }
        self.outputConstr: Optional[Formulation] = None
        self.netModel = None
        self.label = None

    def load(self, propIndex: int, type: str, image=None, label=None):
        if type == "acas":
            inputBounds = getNormaliseInput(propIndex)
            self.inputBounds["ub"] = inputBounds[1]
            self.inputBounds["lb"] = inputBounds[0]
            self.outputConstr = acas_properties[propIndex]["outputConstraints"][-1]
        elif type == "mnist":
            radius = GlobalSetting.img_radius
            self.inputBounds["ub"] = np.minimum(image + radius, 1)
            self.inputBounds["lb"] = np.maximum(image - radius, 0)
            self.label = label

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
