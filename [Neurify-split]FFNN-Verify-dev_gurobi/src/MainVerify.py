from multiprocessing import Queue
from SplittingProcess import SplittingProcess
from LinearFunction import LinearFunction

class MainVerify:

    def __init__(self, network, spec):
        self.globalQueue = Queue()
        initialSplittingProcess = SplittingProcess(0, network.lmodel, spec, self.globalQueue)
        initialProblems = initialSplittingProcess.getInitialProblemSet(processNumber=4)
        self.splittingProcessQueue = [
            SplittingProcess(
                i+1,
                initialProblems[i][0],
                initialProblems[i][1],
                self.globalQueue
            ) for i in range(len(initialProblems))
        ]


    def verify(self):
        pass
