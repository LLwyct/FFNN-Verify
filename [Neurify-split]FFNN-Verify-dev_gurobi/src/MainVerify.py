from Split import Split
from Network import Network
from typing import List
from multiprocessing import Queue
from Specification import Specification
from timeit import default_timer as timer
from SplittingProcess import SplittingProcess
from ModelVerificationProcess import ModelVerificationProcess
from EnumMessageType import EnumMessageType

class MainVerify:

    def __init__(self, network: 'Network', spec: 'Specification'):
        self.jobQueue: 'Queue' = Queue()
        self.messageQueue: 'Queue' = Queue()
        initialSplittingProcess = SplittingProcess("0_0", network.lmodel, spec, self.jobQueue, self.messageQueue)
        initialProblems: List['Split'] = initialSplittingProcess.getInitialProblemSet(processNumber=2)
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
            ) for i in range(2)
        ]

    def verify(self):
        finalSatResult = True
        finalSolverRuntime = 0
        finishedProcessNum = 0
        jobs_num = 0
        solverModelSolvedProblemNums = 0
        start = timer()
        for process in self.splittingProcessQueue:
            process.start()
        for process in self.modelVerificationProcessQueue:
            process.start()

        while True:
            try:
                res = self.messageQueue.get(timeout=3600)
                if res[0] == EnumMessageType.PROCESS_SENT_JOBS_NUM:
                    jobs_num += res[1]
                elif res[0] == EnumMessageType.SPLITTING_PROCESS_FINISHED:
                    finishedProcessNum += 1
                elif res[0] == EnumMessageType.VERIFY_RESULT:
                    jobs_num -= 1
                    solverResult: bool = res[1]
                    solverRuntime: float = res[2]
                    solverModelSolvedProblemNums += 1
                    finalSolverRuntime += solverRuntime
                    finalSatResult = solverResult
                    if solverResult == False:
                        finalSatResult = False
                        break

                if finishedProcessNum == len(self.splittingProcessQueue) and jobs_num == 0:
                    print("all end")
                    break
                print(finishedProcessNum, jobs_num)
            except Exception:
                raise Exception
        end = timer()

        try:
            for process in self.splittingProcessQueue:
                process.terminate()
            for process in self.modelVerificationProcessQueue:
                process.terminate()
        except Exception:
            print("something error while attempting to terminate processes")
            raise Exception

        return (finalSatResult, (end - start), finalSolverRuntime)
