class Options:
    def __init__(
            self,
            path: str = "",
            checkType: str = "acas",
            property: int = 1
    ):
        self.netPath = path
        self.checkType = checkType
        self.property = property

class GlobalSetting:
    # 0 MILP
    # 1 triangle relaxtion
    constrMethod = 0
    # 0 MIL with bigM
    # 1 MILP with ia  区间传播
    # 2 MILP with sia 符号区间传播
    # 3 MILP with slr 符号线性松弛
    # 4 sia and slr
    preSolveMethod = 2
    use_bounds_opt = True