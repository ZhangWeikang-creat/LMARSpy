# Purpose: Atm Utility functions

from lmarspy.model.duogrid import Duogrid
from .data import AtmData

class AtmUtil:

    @staticmethod
    def ext_scalars(atm: AtmData) -> None:

        fps = atm.fps

        fps.exchange_boundary(atm.delp)
        fps.exchange_boundary(atm.delz)
        fps.exchange_boundary(atm.pt)
        fps.exchange_boundary(atm.wa)
        fps.exchange_boundary(atm.ua)
        fps.exchange_boundary(atm.va)

        return


