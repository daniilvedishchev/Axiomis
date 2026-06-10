class TargetBuilderError(Exception):
    pass

class InvalidLagError(TargetBuilderError):
    pass

class MissingColumnError(TargetBuilderError):
    pass