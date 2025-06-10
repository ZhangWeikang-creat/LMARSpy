#!/bin/bash

export PYTHONPATH=/lmarspy:$PYTHONPATH

input_name="input_sw_gauss_1d"
output_name="output_sw_gauss_1d"

cat <<EOL > $input_name.yaml
times:
    YYYY: 2000
    MM: 1
    DD: 1
    days: 0
    hours: 0
    minutes: 0
    seconds: 400
    dt_atmos: 1

dims:
    global_npx: 201
    global_npy: 2
    npz: 1
    ng: 3

fpses:
    px: 1
    py: 1
    
ics:
    ic_type: 11

ios:
    do_output: True
    nc_parallel: True
    out_fre: 10
    do_diag: True

dyns:
    k_split: 1
    n_split: 1
    rk: 333
    dyn_core: eul_sw_linear
    lim: False
    lim_deg: 0
EOL

mpiexec -np 1 python -u /lmarspy/lmarspy/model/main.py --input $input_name.yaml --output $output_name \
        --backend numpy --device cpu

rm -f $input_name.yaml