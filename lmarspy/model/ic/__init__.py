#!/usr/bin/env python

# Purpose: Creating initial conditions for LMARS model
# Author: Chen Research Group (xichen.me@outlook.com)

from .base import ICbase
from .data import ICdata
from lmarspy.model.atm import Atm

from lmarspy.fps.data import FpsData

class IC(ICdata):
    
    def assign_atm(self, fps: FpsData, atm: Atm):

        ICbase.assign_atm(self, fps, atm)

        return
