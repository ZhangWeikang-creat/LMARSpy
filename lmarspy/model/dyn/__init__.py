# Purpose: Dynamical core of LMARS

from .data import DynData
from .ctrl import DynCtrl


class Dyn(DynData):


    def step(self) -> None:

        DynCtrl.step(self)

        return
