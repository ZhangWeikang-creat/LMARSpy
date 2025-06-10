#!/usr/bin/env python

# Purpose: Ctrl
# Author: Chen Research Group (xichen.me@outlook.com)


from .data import CtrlData
from .step import CtrlStep

class Ctrl(CtrlData):


    def step(self: CtrlData, na) -> None:

        CtrlStep.step(self, na)

        return


