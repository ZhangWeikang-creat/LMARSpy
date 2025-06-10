#!/bin/bash

nps=(160 240 320 400)

for np in "${nps[@]}"
do

input_name="input_robert_gauss"
output_name="output_robert_gauss_${np}_numpy_16cpu_nojit"

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
    global_npx: $((np + 1))
    global_npy: $((np + 1))
    npz: $((np/10))
    ng: 3

fpses:
    px: 4
    py: 4

ics:
    ic_type: 23

ios:
    do_output: False
    nc_parallel: True
    out_fre: 1
    do_diag: True

dyns:
    k_split: 2
    n_split: 5
    rk: 333
    dyn_core: eul
    lim: False
    lim_deg: 0
EOL

mpiexec -np 16 lmarspy --input $input_name.yaml --output $output_name \
        --backend numpy --device cpu

rm -f $input_name.yaml

done