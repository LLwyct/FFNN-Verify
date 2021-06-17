from typing import List, Tuple, Union


class Formulation:
    '''
        [
            ("VarVar", "0", "GT", "1"),
            ("VarVar", "0", "GT", "2"),
            ("VarVar", "0", "GT", "3"),
            ("VarVar", "0", "GT", "4"),
        ]
    '''
    def __init__(self) -> None:
        self.constraints: List[Tuple[str, str, str, str]]
        pass

class Disjunctive(Formulation):
    def __init__(self, constrList: List[Tuple[str, str, str, str]]):
        super(Disjunctive, self).__init__()
        self.constraints = constrList

class Conjunctive(Formulation):
    def __init__(self, constrList: List[Tuple[str, str, str, str]]):
        super(Conjunctive, self).__init__()
        self.constraints = constrList
