from lmarspy.backend.field import Field

if Field.backend == "numpy":
    from .cal_numpy import cal
elif Field.backend == "torch":
    from .cal_torch import cal
else:
    raise ValueError(f'Unsurpported backend: {Field.backend}')
