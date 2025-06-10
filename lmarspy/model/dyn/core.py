# Purpose: acoustic time-level LMARS dyn core controller, the most essential level


from .data import DynData
from .eul import DynEul
from .eul_dis import DynEulDis
from .eul_lim import DynEulLim
from .eul_sw import DynEulSw
from .eul_sw_lim import DynEulSwLim
from .eul_sw_linear import DynEulSwLinear
from .eul_sw_linear_lim import DynEulSwLinearLim
from .eul_en import DynEulEn
from .eul_en_dis import DynEulEnDis
from .eul_vic import DynEulVic
from .eul_vic_lim import DynEulVicLim
from .eul_en_vic import DynEulEnVic
class DynCore:

    @staticmethod
    def step(dyn: DynData, dt: float, ng: int = 1, lim: bool = False) -> None:
        
        fg = dyn.fg

        if fg.dyn_core == "eul":
            if not lim:
                DynEul.step(dyn, dt, ng)
            else:
                DynEulLim.step(dyn, dt, ng)

        elif fg.dyn_core == "eul_sw":
            if not lim:
                DynEulSw.step(dyn, dt, ng)
            else:
                DynEulSwLim.step(dyn, dt, ng)
        
        elif fg.dyn_core == "eul_sw_linear":
            if not lim:
                DynEulSwLinear.step(dyn, dt, ng)
            else:
                DynEulSwLinearLim.step(dyn, dt, ng)
        
        elif fg.dyn_core == "eul_dis":
            if not lim:
                DynEulDis.step(dyn, dt, ng)
            else:
                dyn.fps.raise_error(f"There is no dyn core with limiter to select!")

        elif fg.dyn_core == "eul_vic":
            if not lim:
                DynEulVic.step(dyn, dt, ng)
            else:
                DynEulVicLim.step(dyn, dt, ng)

        elif fg.dyn_core == "eul_en":
            if not lim:
                DynEulEn.step(dyn, dt, ng)
            else:
                dyn.fps.raise_error(f"Dyn core of energy can not have limiter!")

        elif fg.dyn_core == "eul_en_dis":
            if not lim:
                DynEulEnDis.step(dyn, dt, ng)
            else:
                dyn.fps.raise_error(f"Dyn core of energy can not have limiter!")
            
        elif fg.dyn_core == "eul_en_vic":
            if not lim:
                DynEulEnVic.step(dyn, dt, ng)
            else:
                dyn.fps.raise_error(f"Dyn core of energy can not have limiter!")
            
        else:
            dyn.fps.raise_error(f"There is no dyn core to select!")

        


