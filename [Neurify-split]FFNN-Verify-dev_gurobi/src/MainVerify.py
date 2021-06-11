from multiprocessing import Queue
from typing import List, Tuple
from LayerModel import LayerModel
from Specification import Specification
from Split import Split
from SplittingProcess import SplittingProcess
from LinearFunction import LinearFunction

class MainVerify:

    def __init__(self, network, spec: Specification):
        self.globalQueue = Queue()
        initialSplittingProcess = SplittingProcess(0, network.lmodel, spec, self.globalQueue)
        initialProblems: List[Split] = initialSplittingProcess.getInitialProblemSet(processNumber=4)
        self.splittingProcessQueue: List[SplittingProcess] = [
            SplittingProcess(
                i+1,
                network.lmodel,
                spec.resetFromSplit(initialProblems[i]),
                self.globalQueue
            ) for i in range(len(initialProblems))
        ]
        

    def verify(self):
        for process in self.splittingProcessQueue:
            process.run()