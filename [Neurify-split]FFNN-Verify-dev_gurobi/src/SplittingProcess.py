import copy
from Split import Split
from collections import deque
from options import GlobalSetting
from LayerModel import LayerModel
from VerifyModel import VerifyModel
from Specification import Specification
from multiprocessing import Process, Queue
from ConstraintFormula import Disjunctive
from typing import NoReturn, Tuple, List
from EnumMessageType import EnumMessageType

class SplittingProcess(Process):
    def __init__(self, id: str, lmodel: 'LayerModel', spec: 'Specification', globalJobQueue, globalMsgQueue):
        super(SplittingProcess, self).__init__()
        self.id: str = id
        self.lmodel: 'LayerModel' = lmodel
        self.spec: 'Specification' = spec
        self.globalJobQueue: 'Queue' = globalJobQueue
        self.globalMsgQueue: 'Queue' = globalMsgQueue

    def run(self) -> NoReturn:
        try:
            self.findSubProblemsAndPushToGlobalQueue()
        except Exception:
            raise Exception

    def getInitialProblemSet(self, processNumber=1) -> List['Split']:
        split = Split.createByInterval("0_0", self.spec)
        # initialVerifyModel, initialFixedRatio = self.getFixedNodeInfo(self.lmodel, split)
        splits: List['Split'] = [split]

        while len(splits) < processNumber:
            nextLoopSplit = []
            for split in splits:
                subProblems, _ = self.getBestSplit(split, self.lmodel, self.spec)
                nextLoopSplit.extend(subProblems)
            splits = nextLoopSplit

        return splits

    def getFixedNodeInfo(self, lmodel: 'LayerModel', split: 'Split') -> Tuple['VerifyModel', float]:
        verifyModel = VerifyModel(split.id, lmodel, self.spec.resetFromSplit(split))
        # 在执行这一步之前，verifyModel中的lmodel还未完成presolve，只有输入的区间
        verifyModel.initAllBounds()
        return verifyModel, verifyModel.getFixedNodeRatio()

    def getBestSplit(self, split: 'Split', lmodel, spec) -> Tuple[List['Split'], int]:
        # 如果输入维度小于某个值才执行逐维度测试
        inputSize = len(split.up)
        if inputSize < 15:
            bestIndex = -1
            bestAverageRatio = 0
            bestSplit = None
            for i in range(inputSize):
                new_1_upper = split.up.copy()
                new_2_upper = split.up.copy()
                new_1_lower = split.lo.copy()
                new_2_lower = split.lo.copy()
                middleValue = (split.up[i] + split.lo[i]) / 2
                new_1_upper[i] = middleValue
                new_2_lower[i] = middleValue
                newSpec1: 'Specification' = copy.deepcopy(spec)
                newSpec2: 'Specification' = copy.deepcopy(spec)
                newSpec1.setInputBounds(new_1_upper, new_1_lower)
                newSpec2.setInputBounds(new_2_upper, new_2_lower)

                verifyModel1 = VerifyModel("0_0", lmodel, newSpec1)
                verifyModel2 = VerifyModel("0_0", lmodel, newSpec2)
                verifyModel1.initAllBounds()
                verifyModel2.initAllBounds()

                ratio1 = verifyModel1.getFixedNodeRatio()
                ratio2 = verifyModel2.getFixedNodeRatio()
                averageRatio = (ratio1 + ratio2) / 2

                if averageRatio > bestAverageRatio:
                    bestAverageRatio = averageRatio
                    bestIndex = i
                    [group, seq] = split.id.split('_')
                    bestSplit = [
                        Split("{}_{}".format(str(int(group)), str(int(seq)*2)), new_1_upper, new_1_lower),
                        Split("{}_{}".format(str(int(group)), str(int(seq)*2+1)), new_2_upper, new_2_lower)
                    ]

            return bestSplit, bestIndex
        else:
            pass

    def findSubProblemsAndPushToGlobalQueue(self) -> NoReturn:
        topSplit: 'Split' = Split.createByInterval(self.id, self.spec)
        # topVerifyModel, topFixedRatio = self.getFixedNodeInfo(self.lmodel, topSplit)
        queue = deque()
        queue.append(topSplit)
        processSentJobsNum: int = 0
        while(len(queue) != 0):
            nowSplit: 'Split' = queue.pop()
            worth, subSplit = self.isSplitWorth(nowSplit)
            if worth:
                print("push", subSplit[0].id, subSplit[1].id)
                queue.extend(subSplit)
            elif isinstance(subSplit, str):
                print("safe", subSplit)
                continue
            elif isinstance(subSplit, VerifyModel):
                print("slover", subSplit.id)
                self.globalJobQueue.put(subSplit)
                processSentJobsNum += 1
        self.globalMsgQueue.put((EnumMessageType.PROCESS_SENT_JOBS_NUM, processSentJobsNum))
        self.globalMsgQueue.put(tuple([EnumMessageType.SPLITTING_PROCESS_FINISHED]))

    def isSplitWorth(self, nowSplit: 'Split') -> Tuple:
        nowVerifyModel, nowFixedRatio = self.getFixedNodeInfo(self.lmodel, nowSplit)
        # 如果该split是安全的，则该split及其子节点都是安全的
        if self.isSatisfySpec(nowVerifyModel):
            return False, nowSplit.id
        # 如果该split导致其gurobi的节点固定率很高，则没必要继续分割
        if nowFixedRatio > GlobalSetting.SPLIT_THRESHOLD:
            return False, nowVerifyModel
        
        # 否则该split还可以继续分割
        subsplit, bestSplitIndex = self.getBestSplit(nowSplit, self.lmodel, self.spec.resetFromSplit(nowSplit))
        vmodel0, ratio0 = self.getFixedNodeInfo(self.lmodel, subsplit[0])
        vmodel1, ratio1 = self.getFixedNodeInfo(self.lmodel, subsplit[1])
        if nowFixedRatio > max(ratio0, ratio1):
            return False, nowVerifyModel
        
        return True, subsplit

    def isSatisfySpec(self, vmodel: 'VerifyModel'):
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

        return ans
