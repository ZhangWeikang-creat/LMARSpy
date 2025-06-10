#!/usr/bin/env python

# Purpose: physics laws, use "sci" to reserve "phy" for physics parameterizations.
# Author: Chen Research Group (xichen.me@outlook.com)


from lmarspy.backend.field import Field

if Field.backend == "numpy":
    from .sci_numpy import Sci
elif Field.backend == "torch":
    from .sci_torch import Sci
else:
    raise ValueError(f'Unsurpported backend: {Field.backend}')
