from multiprocessing import Process
from LayerModel import LayerModel
from Specification import Specification
from VerifyModel import VerifyModel
from typing import NoReturn, Tuple, List
from Specification import Specification
import copy
from collections import deque
from numpy import ndarray
from Split import Split
from ConstraintFormula import Disjunctive


class SplittingProcess(Process):
    def __init__(self, id: int, lmodel: LayerModel, spec: Specification, queue):
        super().__init__()
        self.id: int = id
        self.lmodel = lmodel
        self.spec: Specification = spec
        self.queue = queue

    def run(self) -> NoReturn:
        self.findSubProblemsAndPushToGlobalQueue()

    def getInitialProblemSet(self, processNumber=2) -> List[Split]:
        split = Split.createByInterval(self.spec)
        # initialVerifyModel, initialFixedRatio = self.getFixedNodeInfo(self.lmodel, split)
        splits: List[Split] = [split]

        while len(splits) <= processNumber:
            nextLoopSplit = []
            for split in splits:
                subProblems, _ = self.getBestSplit(split, self.lmodel, self.spec)
                nextLoopSplit.extend(subProblems)
            splits = nextLoopSplit

        return splits

    def getFixedNodeInfo(self, lmodel: LayerModel, split: Split) -> Tuple[VerifyModel, float]:
        verifyModel = VerifyModel(lmodel, self.spec.resetFromSplit(split))
        # 在执行这一步之前，verifyModel中的lmodel还未完成presolve，只有输入的区间
        verifyModel.initAllBounds()
        return verifyModel, verifyModel.getFixedNodeRatio()

    def getBestSplit(self, split: Split, lmodel, spec) -> Tuple[List[Split], int]:
        # 如果输入维度小于某个值才执行逐维度测试
        inputSize = len(split.up)
        if inputSize < 15:
            bestIndex = -1
            bestAverageRatio = 0
            bestSplit = None
            for i in range(inputSize):
                new_1_upper = split.left.copy()
                new_2_upper = split.left.copy()
                new_1_lower = split.right.copy()
                new_2_lower = split.right.copy()
                middleValue = (split.up[i] + split.lo[i]) / 2
                new_1_upper[i] = middleValue
                new_2_lower[i] = middleValue
                newSpec1: Specification = copy.deepcopy(spec)
                newSpec2: Specification = copy.deepcopy(spec)
                newSpec1.setInputBounds(new_1_upper, new_1_lower)
                newSpec2.setInputBounds(new_2_upper, new_2_lower)

                verifyModel1 = VerifyModel(lmodel, newSpec1)
                verifyModel2 = VerifyModel(lmodel, newSpec2)
                verifyModel1.initAllBounds()
                verifyModel2.initAllBounds()

                ratio1 = verifyModel1.getFixedNodeRatio()
                ratio2 = verifyModel2.getFixedNodeRatio()
                averageRatio = (ratio1 + ratio2) / 2

                if averageRatio > bestAverageRatio:
                    bestAverageRatio = averageRatio
                    bestIndex = i
                    bestSplit = [Split(split.id*2, new_1_upper, new_1_lower), Split(split.id*2+1, new_2_upper, new_2_lower, new_2_lower)]

            return bestSplit, bestIndex
        else:
            pass

    def findSubProblemsAndPushToGlobalQueue(self):
        topSplit = Split.createByInterval(self.spec)
        topVerifyModel, topFixedRatio = self.getFixedNodeInfo(self.lmodel, topSplit)
        queue = deque()
        queue.append(topSplit)

        while(len(queue) != 0):
            split: Split = queue.pop()
            worth, subSplit = self.isSplitWorth(split, topVerifyModel, topFixedRatio)

    def isSplitWorth(self, split: Split, vmodel: VerifyModel, topFixedRatio: float):
        if not self.isSatisfySpec(vmodel):
            return False, []
        '''
        TODO
        '''

    def isSatisfySpec(self, vmodel: VerifyModel):
        upper = vmodel.lmodel.lmodels[-1].var_bounds_out["ub"]
        lower = vmodel.lmodel.lmodels[-1].var_bounds_out["lb"]
        ans = True
        if isinstance(self.spec.outputConstr, Disjunctive):
            constrs = self.spec.outputConstr.constraints
            for constr in constrs:
                relation = constr[2]
                if constr[0] == "VarVar":
                    left = int(constr[1])
                    right = int(constr[3])
                    if relation == "GT":
                        if not lower[left] >= upper[right]:
                            ans = False
                            break
                    elif relation == "LT":
                        if not upper[left] <= lower[right]:
                            ans = False
                            break
                    elif relation == "EQ":
                        pass
                    else:
                        raise Exception("输出约束关系异常")
                elif constr[0] == "VarValue":
                    var = int(constr[1])
                    value = float(constr[3])
                    if relation == "GT":
                        if not lower[var] >= value:
                            ans = False
                            break
                    elif relation == "LT":
                        if not upper[var] <= value:
                            ans = False
                            break
                    elif relation == "EQ":
                        pass
                    else:
                        raise Exception("输出约束关系异常")

        return False