#!/bin/bash

input_name="input_acoustic_wave_damping"
output_name="output_acoustic_wave_damping"

cat <<EOL > $input_name.yaml
times:
    YYYY: 2000
    MM: 1
    DD: 1
    days: 0
    hours: 0
    minutes: 0
    seconds: 2
    dt_atmos: 1

dims:
    global_npx: 2
    global_npy: 2
    npz: 680
    ng: 3
    
fpses:
    px: 1
    py: 1

ics:
    ic_type: 31

ios:
    do_output: True
    nc_parallel: True
    out_fre: 1
    do_diag: True

dyns:
    k_split: 10
    n_split: 34
    rk: 333
    dyn_core: eul_vic
    lim: False
    lim_deg: 0
EOL

mpiexec -np 1 lmarspy --input $input_name.yaml --output $output_name \
        --backend numpy --device cpu
        
rm -f $input_name.yaml