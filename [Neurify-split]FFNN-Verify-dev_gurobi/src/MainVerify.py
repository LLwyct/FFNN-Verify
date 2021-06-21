from Split import Split
from Network import Network
from typing import List
from multiprocessing import Queue
from Specification import Specification
from options import GlobalSetting
from SplittingProcess import SplittingProcess
from ModelVerificationProcess import ModelVerificationProcess
from EnumMessageType import EnumMessageType

class MainVerify:

    def __init__(self, network: 'Network', spec: 'Specification'):
        self.jobQueue: 'Queue' = Queue()
        self.messageQueue: 'Queue' = Queue()
        initialSplittingProcess = SplittingProcess("0_0", network.lmodel, spec, self.jobQueue, self.messageQueue)
        initialProblems: List['Split'] = initialSplittingProcess.getInitialProblemSet(processNumber=GlobalSetting.splitting_processes_num)
        self.splittingProcessQueue: List['SplittingProcess'] = [
            SplittingProcess(
                "{}_1".format(i+1),
                network.lmodel,
                spec.resetFromSplit(initialProblems[i]),
                self.jobQueue,
                self.messageQueue
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

        jobs_num:                       int   = 0        # splitting进程向solver进程提交的总作业数
        finalSatResult:                 bool  = True     # 网络最终结果sat or unsat
        solverProcessRuntime:           float = 0        # 求解器运算阶段总时长
        splitProcessTotalRuntime:       float = 0        # splitting阶段总时长
        finishedSplittingProcessNum:    int   = 0        # 完成的split进程数
        solverModelSolvedProblemNums:   int   = 0
        for process in self.splittingProcessQueue:
            process.start()
        for process in self.modelVerificationProcessQueue:
            process.start()

        while True:
            try:
                res = self.messageQueue.get(timeout=3600)
                if res[0] == EnumMessageType.SPLITTING_PROCESS_FINISHED:
                    '''
                    res[1] 代表 该子进程向solver提交求解split数目
                    res[2] 代表 该进程在splitting阶段花费总时长
                    '''
                    jobs_num += res[1]
                    splitProcessTotalRuntime += res[2]
                    finishedSplittingProcessNum += 1
                elif res[0] == EnumMessageType.VERIFY_RESULT:
                    '''
                    res[1] 代表 该split对应的vmodel的求解结果，True表示sat，False表示unsat
                    res[2] 代表 一个split在solver中的求解时长
                    '''
                    jobs_num -= 1
                    solverResult: bool = res[1]
                    solverRuntime: float = res[2]
                    solverModelSolvedProblemNums += 1
                    solverProcessRuntime += solverRuntime
                    finalSatResult = solverResult
                    if solverResult == False:
                        finalSatResult = False
                        break

                if finishedSplittingProcessNum == len(self.splittingProcessQueue) and jobs_num == 0:
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

        
        return (finalSatResult, splitProcessTotalRuntime, solverProcessRuntime)
