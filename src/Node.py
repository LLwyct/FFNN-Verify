from enum import Enum

class NodeType(Enum):
    CONTINUE:   0
    RELU:       1
    ELSE:       2


class Node:
    def __init__(self, gvar, ):
        self.gvar = None
        self.type: NodeType = NodeType.ELSE
        self.ub = None
        self.lb = None

