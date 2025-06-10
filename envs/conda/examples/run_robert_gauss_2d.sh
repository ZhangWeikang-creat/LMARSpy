#!/bin/bash

input_name="input_robert_gauss_2d"
output_name="output_robert_gauss_2d"

cat <<EOL > $input_name.yaml
times:
    YYYY: 2000
    MM: 1
    DD: 1
    days: 0
    hours: 0
    minutes: 0
    seconds: 1080
    dt_atmos: 1 

dims:
    global_npx: 201
    global_npy: 2
    npz: 300
    ng: 3

fpses:
    px: 4
    py: 1
    
ics:
    ic_type: 21

ios:
    do_output: True
    nc_parallel: True
    out_fre: 10
    do_diag: True

dyns:
    k_split: 5
    n_split: 25
    rk: 333
    dyn_core: eul
    lim: False
    lim_deg: 0
EOL

mpiexec -np 4 lmarspy --input $input_name.yaml --output $output_name \
        --backend numpy --device cpu

rm -f $input_name.yaml