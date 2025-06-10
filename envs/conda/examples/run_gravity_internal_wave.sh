#!/bin/bash

input_name="input_gravity_internal_wave"
output_name="output_gravity_internal_wave"

cat <<EOL > $input_name.yaml
times:
    YYYY: 2000
    MM: 1
    DD: 1
    days: 0
    hours: 0
    minutes: 0
    seconds: 3000
    dt_atmos: 100

dims:
    global_npx: 301
    global_npy: 2
    npz: 100
    ng: 3
    
fpses:
    px: 4
    py: 1

ics:
    ic_type: 52

ios:
    do_output: True
    nc_parallel: True
    out_fre: 1
    do_diag: True

dyns:
    k_split: 5
    n_split: 10
    rk: 333
    dyn_core: eul_vic
    lim: False
    lim_deg: 0
EOL

mpiexec -np 4 lmarspy --input $input_name.yaml --output $output_name \
        --backend numpy --device cpu

rm -f $input_name.yaml