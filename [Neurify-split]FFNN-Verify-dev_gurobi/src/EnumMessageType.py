from enum import Enum

class EnumMessageType(Enum):
    PROCESS_SENT_JOBS_NUM = 0
    SPLITTING_PROCESS_FINISHED = 1
    VERIFY_RESULT = 2
