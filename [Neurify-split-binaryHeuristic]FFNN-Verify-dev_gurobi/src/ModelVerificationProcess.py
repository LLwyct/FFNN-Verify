from VerifyModel import VerifyModel
from timeit import default_timer as timer
from multiprocessing import Process, Queue
from EnumMessageType import EnumMessageType

class ModelVerificationProcess(Process):
    def __init__(self, id: int, globalJobQueue: 'Queue', globalMsgQueue: 'Queue', netModel=None) -> None:
        super(ModelVerificationProcess, self).__init__()
        self.id: int = id
        self.globalJobQueue: 'Queue' = globalJobQueue
        self.globalMsgQueue: 'Queue' = globalMsgQueue
        self.networkModel = netModel

    def run(self) -> None:
        while True:
            try:
                # 当自时隔1h依旧从队列取不出对象时则视为verifyModel生产队列为空，结束进程
                verifyModel: 'VerifyModel' = self.globalJobQueue.get(timeout=3600)
                startTime = timer()
                res: bool = verifyModel.verify(self.networkModel)
                endTime = timer()
                self.globalMsgQueue.put((EnumMessageType.VERIFY_RESULT, res, endTime - startTime, verifyModel.id))
                if res == False:
                    break
            except Exception:
                raise Exception
