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

class GlobalSettingClass:
    # 0 MILP
    # 1 triangle relaxtion
    constrMethod = 0
    # 0 MIL with bigM
    # 1 MILP with ia  区间传播
    # 2 MILP with sia 符号区间传播
    # 3 MILP with slr 符号线性松弛
    # 4 sia and slr
    preSolveMethod = 4
    use_bounds_opt = True
    SPLIT_THRESHOLD = 0
    # splitting_processes_num应该取2^k(k = 0,1,2,3...)
    splitting_processes_num = 0
    vmodel_verify_processes_num = 1
    DEBUG_MODE = False
    use_binary_heuristic_method = True
    write_to_file = False
    img_radius = 0.05
    TIME_OUT = 100
    def __init__(self):
        self.networkModel = None

    def setNetworkModel(self, networkModel):
        self.networkModel = networkModel

GlobalSetting = GlobalSettingClass()
