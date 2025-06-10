#!/usr/bin/env python

# Purpose: computational grid for LMARS
# Author: Chen Research Group (xichen.me@outlook.com)

from lmarspy.backend.field import Field
from lmarspy.backend.cal import cal
from lmarspy.model.mp import Mp

from .data import DuogridData
from .util import DuogridUtil


class Duogrid(DuogridData):

    def assign_cart_metrics(self) -> None:

        DuogridUtil.assign_cart_metrics(self)

        return


    @staticmethod
    def ext_scalar(var: Field, mp:Mp) -> None:

        DuogridUtil.ext_scalar(var, mp)

        return
    
    @staticmethod
    def ext_vector(ua: Field, va: Field, mp:Mp) -> None:

        DuogridUtil.ext_vector(ua, va, mp)

        return



