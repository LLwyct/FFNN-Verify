import copy
from enum import Enum
from Split import Split
from collections import deque
from options import GlobalSetting
from LayerModel import LayerModel
from VerifyModel import VerifyModel
from Specification import Specification
from typing import NoReturn, Tuple, List
from timeit import default_timer as timer
from multiprocessing import Process, Queue
from ConstraintFormula import Disjunctive
from EnumMessageType import EnumMessageType

class SplitEndReason:
    SPLIT_SAFETY        = 0
    SPLIT_SAT_THRESHOLD = 1
    SPLIT_CANT_BETTER   = 2
    CONTINUE            = 3


class SplittingProcess(Process):
    def __init__(self, processId: str, topSplitId: str, lmodel: 'LayerModel', spec: 'Specification', globalJobQueue, globalMsgQueue, globalInfoQueue):
        super(SplittingProcess, self).__init__()
        self.id: str                  = processId
        self.topSplitId               = topSplitId
        self.lmodel: 'LayerModel'     = lmodel
        self.spec: 'Specification'    = spec
        self.globalJobQueue: 'Queue'  = globalJobQueue
        self.globalMsgQueue: 'Queue'  = globalMsgQueue
        self.globalInfoQueue: 'Queue' = globalInfoQueue

    def run(self) -> NoReturn:
        try:
            self.findSubProblemsAndPushToGlobalQueue()
        except Exception:
            raise Exception

    def findSubProblemsAndPushToGlobalQueue(self) -> NoReturn:
        start = timer()
        topSplit: 'Split' = Split.createByInterval(self.topSplitId, self.spec)
        # topVerifyModel, topFixedRatio = self.getFixedNodeInfo(self.lmodel, topSplit)
        queue = deque()
        queue.append(topSplit)
        processSentJobsNum: int = 0
        while(len(queue) != 0):
            nowSplit: 'Split' = queue.pop()
            self.globalInfoQueue.put(("push", nowSplit.id))
            # 这里的subsplit就已经有做好区间传播的lmodel模型了
            worth, reason, subSplit = self.isSplitWorth(nowSplit)
            if worth:
                self.globalInfoQueue.put(("extend", nowSplit.id))
                queue.extend(subSplit)
                continue
            if reason == SplitEndReason.SPLIT_SAFETY:
                self.globalInfoQueue.put(("safe", nowSplit.id))
                assert isinstance(subSplit, str)
                continue
            if reason == SplitEndReason.SPLIT_SAT_THRESHOLD or reason == SplitEndReason.SPLIT_CANT_BETTER:
                '''
                如果worth为False，那么有三种可能，第一种是这个节点本身是安全的，没有必要再往下做
                第二种是由于input分割致使该模型的relu节点fixed比率相当高，达到我们规定的阈值，则没有必要再分割
                第三种是由于该节点的下一次分割的两个字模型的固定比率虽然没有达到阈值，但是继续分割没有提升，则没有必要再分割
                '''
                self.globalInfoQueue.put(("solve", nowSplit.id))
                assert isinstance(subSplit, VerifyModel)
                self.globalJobQueue.put(subSplit)
                processSentJobsNum += 1
        end = timer()
        self.globalMsgQueue.put((EnumMessageType.SPLITTING_PROCESS_FINISHED, processSentJobsNum, end - start))

    def getInitialProblemSet(self, processNumber=1) -> List['Split']:
        split = Split.createByInterval("1", self.spec)
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
        bestIndex = -1
        bestAverageRatio = 0
        bestSplit = None
        if inputSize < 15:
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
                    id = split.id
                    bestSplit = [
                        Split("{}".format(str(int(id)*2)), new_1_upper, new_1_lower),
                        Split("{}".format(str(int(id)*2 + 1)), new_2_upper, new_2_lower)
                    ]
            return bestSplit, bestIndex
        else:
            max_interval = 0
            max_interval_index = 0
            for i in range(len(split.up)):
                if split.up[i] - split.lo[i] > max_interval and split.up[i] != 1 and split.up[i] != 0:
                    max_interval_index = i
                    max_interval = split.up[i] - split.lo[i]
            new_1_upper = split.up.copy()
            new_2_upper = split.up.copy()
            new_1_lower = split.lo.copy()
            new_2_lower = split.lo.copy()
            middleValue = (split.up[max_interval_index] + split.lo[max_interval_index]) / 2
            new_1_upper[max_interval_index] = middleValue
            new_2_lower[max_interval_index] = middleValue
            id = split.id
            return [
                    Split("{}".format(str(int(id) * 2)), new_1_upper, new_1_lower),
                    Split("{}".format(str(int(id) * 2 + 1)), new_2_upper, new_2_lower)
            ], max_interval_index

    def isSplitWorth(self, nowSplit: 'Split') -> Tuple[bool, int, any]:
        '''
        False SplitEndReason.SPLIT_SAFETY           Split.id: str\n
        False SplitEndReason.SPLIT_SAT_THRESHOLD    VerifyModel\n
        False SplitEndReason.SPLIT_CANT_BETTER      VerifyModel\n
        True  SplitEndReason.CONTINUE               subsplit: List[Split]
        '''
        # 获得当前split的已固定节点ratio
        nowVerifyModel, nowFixedRatio = self.getFixedNodeInfo(self.lmodel, nowSplit)
        # 如果该split是安全的，则该split及其子节点都是安全的
        if self.isSatisfySpec(nowVerifyModel):
            return (False, SplitEndReason.SPLIT_SAFETY, nowSplit.id)
        # 如果该split导致其gurobi的节点固定率很高，则没必要继续分割
        if nowFixedRatio > GlobalSetting.SPLIT_THRESHOLD:
            return (False, SplitEndReason.SPLIT_SAT_THRESHOLD, nowVerifyModel)
        
        # 否则该split还可以继续分割
        subsplit, bestSplitIndex = self.getBestSplit(nowSplit, self.lmodel, self.spec.resetFromSplit(nowSplit))
        vmodel0, ratio0 = self.getFixedNodeInfo(self.lmodel, subsplit[0])
        vmodel1, ratio1 = self.getFixedNodeInfo(self.lmodel, subsplit[1])
        # print(nowFixedRatio, ratio1, ratio0)
        # 如果当前节点的分数，比他分割的两个子问题分数最大值还大，则没必要继续分割
        # nowFixedRatio > max(ratio0, ratio1) 如果acas的性能有影响有可能是因为这句话被改了
        if nowFixedRatio >= max(ratio0, ratio1):
            return (False, SplitEndReason.SPLIT_CANT_BETTER, nowVerifyModel)
        
        return (True, SplitEndReason.CONTINUE, subsplit)

    def isSatisfySpec(self, vmodel: 'VerifyModel'):
        upper = vmodel.lmodel.lmodels[-1].var_bounds_out["ub"]
        lower = vmodel.lmodel.lmodels[-1].var_bounds_out["lb"]
        ans = True
        if self.spec.verifyType == "mnist":
            label = self.spec.label
            for i in range(0, 10):
                if i == label:
                    continue
                if lower[label] < upper[i]:
                    return False
        elif self.spec.verifyType == "acas":
            if isinstance(self.spec.outputConstr, Disjunctive):
                constrs = self.spec.outputConstr.constraints
                for constr in constrs:
                    relation = constr[2]
                    if constr[0] == "VarVar":
                        left = int(constr[1][1:])
                        right = int(constr[3][1:])
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
                        var = int(constr[1][1:])
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
