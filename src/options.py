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
