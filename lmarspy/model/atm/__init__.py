#!/usr/bin/env python

# Purpose: atm
# Author: Chen Research Group (xichen.me@outlook.com)

from .data import AtmData
from .io import AtmIO
from .util import AtmUtil
from .diag import AtmDiag

class Atm(AtmData):

    def ext_scalars(self):

        AtmUtil.ext_scalars(self)

        return


    def create_file_global(self):

        AtmIO.create_file_global(self)

        return

    def create_file_global_parallel(self):

        AtmIO.create_file_global_parallel(self)

        return
    
    def write_data_parallel(self, tid: int):

        AtmIO.write_data_parallel(self, tid)

        return
    
    def create_file(self):

        AtmIO.create_file(self)

        return
    
    def write_data(self, tid: int):

        AtmIO.write_data(self, tid)

        return
    
    def merge_ncfile(self):

        AtmIO.merge_ncfile(self)

        return
    
    def check_var(self):

        AtmDiag.check_var(self)

    def check_var0(self):

        AtmDiag.check_var0(self)


