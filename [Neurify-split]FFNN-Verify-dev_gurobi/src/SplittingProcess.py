from multiprocessing import Process
from LayerModel import LayerModel
from Specification import Specification
from VerifyModel import VerifyModel
from typing import Tuple, List
from Specification import Specification
import copy

class SplittingProcess(Process):
    def __init__(self, id, lmodel, spec, queue):
        super().__init__()
        self.id = id
        self.lmodel = lmodel
        self.spec = spec
        self.queue = queue

    def run(self):
        pass

    def getInitialProblemSet(self, processNumber=2):
        initialProblemSet = []
        initialVerifyModel, initialFixedRatio = self.getFixedNodeInfo(self.lmodel, self.spec)

        inputSplit: List[Tuple[List, List]] = [self.spec.getInputBounds()]

        while True:
            nextLoopSplit = []
            for split in inputSplit:
                subProblems = self.getBestSplit(split, self.lmodel, self.spec)
                nextLoopSplit.extend(subProblems)
            inputSplit = nextLoopSplit
            if len(inputSplit) >= processNumber:
                break
        
        for split in inputSplit:
            initialProblemSet.append(
                (
                    self.lmodel,
                    Specification(split[0], split[1])
                )
            )
        return initialProblemSet

    def getFixedNodeInfo(self, lmodel: LayerModel, spec: Specification):
        verifyModel = VerifyModel(lmodel, spec)
        # 在执行这一步之前，verifyModel中的lmodel还未完成presolve，只有输入的区间
        verifyModel.initAllBounds()
        return verifyModel, verifyModel.getFixedNodeRatio()

    def getBestSplit(self, split, lmodel, spec):
        # 如果输入维度小于某个值才执行逐维度测试
        inputSize = len(split[0])
        if inputSize < 15:
            # bestIndex = -1
            bestAverageRatio = 0
            bestSplit = None
            for i in range(inputSize):
                new_1_upper = split[0].copy()
                new_2_upper = split[0].copy()
                new_1_lower = split[1].copy()
                new_2_lower = split[1].copy()
                middleValue = (split[0][i] + split[1][i]) / 2
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
                    # bestIndex = i
                    bestSplit = [(new_1_upper, new_1_lower), (new_2_upper, new_2_lower)]

            return bestSplit
        else:
            pass
