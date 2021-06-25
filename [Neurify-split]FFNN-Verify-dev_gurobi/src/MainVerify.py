from Split import Split
from Network import Network
from typing import List, Tuple
from multiprocessing import Queue
from Specification import Specification
from options import GlobalSetting
from SplittingProcess import SplittingProcess
from ModelVerificationProcess import ModelVerificationProcess
from EnumMessageType import EnumMessageType
from operator import itemgetter
from graphviz import Digraph

class MainVerify:

    def __init__(self, network: 'Network', spec: 'Specification'):
        self.jobQueue: 'Queue' = Queue()
        self.infoQueue: 'Queue' = Queue()
        self.messageQueue: 'Queue' = Queue()
        initialSplittingProcess = SplittingProcess("0", "1", network.lmodel, spec, self.jobQueue, self.messageQueue, self.infoQueue)
        initialProblems: List['Split'] = initialSplittingProcess.getInitialProblemSet(processNumber=GlobalSetting.splitting_processes_num)
        self.splittingProcessQueue: List['SplittingProcess'] = [
            SplittingProcess(
                "{}".format(i+1),
                initialProblems[i].id,
                network.lmodel,
                spec.resetFromSplit(initialProblems[i]),
                self.jobQueue,
                self.messageQueue,
                self.infoQueue
            ) for i in range(len(initialProblems))
        ]
        
        self.modelVerificationProcessQueue: List['ModelVerificationProcess'] = [
            ModelVerificationProcess(
                i+1,
                self.jobQueue,
                self.messageQueue,
            ) for i in range(GlobalSetting.vmodel_verify_processes_num)
        ]

    def verify(self):

        cur_jobs_num:                   int   = 0        # splitting进程向solver进程提交的当前作业数
        max_jobs_num:                   int   = 0        # splitting进程向solver进程提交的总作业数
        finalSatResult:                 bool  = True     # 网络最终结果sat or unsat
        solverProcessRuntime:           float = 0        # 求解器运算阶段总时长
        splitProcessTotalRuntime:       float = 0        # splitting阶段总时长
        finishedSplittingProcessNum:    int   = 0        # 完成的split进程数
        solverModelSolvedProblemNums:   int   = 0
        for process in self.splittingProcessQueue:
            process.start()
        for process in self.modelVerificationProcessQueue:
            process.start()

        m = dict()
        while True:
            try:
                res = self.messageQueue.get(timeout=3600)
                if res[0] == EnumMessageType.SPLITTING_PROCESS_FINISHED:
                    '''
                    res[1] 代表 该子进程向solver提交求解split数目
                    res[2] 代表 该进程在splitting阶段花费总时长
                    '''
                    cur_jobs_num += res[1]
                    max_jobs_num  = cur_jobs_num
                    splitProcessTotalRuntime += res[2]
                    finishedSplittingProcessNum += 1
                elif res[0] == EnumMessageType.VERIFY_RESULT:
                    '''
                    res[1] 代表 该split对应的vmodel的求解结果，True表示sat，False表示unsat
                    res[2] 代表 一个split在solver中的求解时长
                    res[3] 代表 id
                    '''
                    cur_jobs_num -= 1
                    solverResult: bool = res[1]
                    solverRuntime: float = res[2]
                    solverModelSolvedProblemNums += 1
                    solverProcessRuntime += solverRuntime
                    finalSatResult = solverResult
                    if solverResult == False:
                        finalSatResult = False
                        m[str(res[3])] = 4
                        break
                    else:
                        m[str(res[3])] = 5

                if finishedSplittingProcessNum == len(self.splittingProcessQueue) and cur_jobs_num == 0:
                    print("all end")
                    break
            except Exception:
                raise Exception

        try:
            for process in self.splittingProcessQueue:
                process.terminate()
            for process in self.modelVerificationProcessQueue:
                process.terminate()
        except Exception:
            print("something error while attempting to terminate processes")
            raise Exception
        
        order: List[Tuple] = []
        while True:
            try:
                res = self.infoQueue.get(timeout=5)
                '''
                0 代表第一次加入树节点
                1 代表该节点已经结束，由于extend
                2 代表该节点已经结束，由于safe
                3 代表该节点已经结束，由于被solve
                '''
                # print('recive', res[1])
                if res[0] == "push":
                    order.append((int(res[1]), 0))
                elif res[0] == "extend":
                    order.append((int(res[1]), 1))
                elif res[0] == "safe":
                    order.append((int(res[1]), 2))
                elif res[0] == "solve":
                    order.append((int(res[1]), m[res[1]]))
            except Exception:
                break

        order = sorted(order, key=itemgetter(0, 1))
        for i in order:
            print(i)

        dot = Digraph()
        nodeSet = set()
        for i in order:
            nodeSet.add(str(i[0]))
        for i in range(1, len(order), 2):
            if order[i][1] == 1:
                dot.node(str(order[i][0]))
            elif order[i][1] == 2:
                dot.node(str(order[i][0]), _attributes={'color': 'green', 'style': 'filled'})
            elif order[i][1] == 4:
                dot.node(str(order[i][0]), _attributes={'color': 'red', 'style': 'filled'})
            elif order[i][1] == 5:
                dot.node(str(order[i][0]), _attributes={'color': 'yellow', 'style': 'filled'})
        for i in nodeSet:
            if str(int(i)*2) in nodeSet:
                dot.edge(i, str(int(i)*2))
            if str(int(i)*2 + 1) in nodeSet:
                dot.edge(i, str(int(i)*2 + 1))
        dot.render('graph/graph')
        
        return (finalSatResult, splitProcessTotalRuntime, solverProcessRuntime, max_jobs_num)
