# lmarspy Introduction

lmarspy is a nonhydrostatic atmospheric model dynamical core that features:

* Apply monotonicity limiter in discontinuous regions.
* Increase the number of CFL by vertical implicit solver.
* Allow high-performance computing in Python on GPUs.

## How to Run?

### 1.Enter the lmarspy directory and create a Conda environment

```bash
cd /path/to/lmarspy
conda env create -f env.yaml -n lmarspy
```

or

```bash
conda env create -n lmarspy
pip install -r requirements.txt
```

### 2.Activate the Conda environment and install lmarspy

```bash
conda activate lmarspy
python -m pip install -e .
```

### 3.Enter the work directory and run the model

```bash
cd /path/to/work
mpiexec -np [1] lmarspy --input [input.yaml] --output [output] --backend [torch]  --device [cpu]
```

* `-np` specifies the number of MPI processes.
* `--input` specifies the input file; details are provided below.
* `--output` specifies the output file (".nc" and ".log") path, default is `output`.
* `--backend` selects the calculation backend, such as `numpy`, `torch`.
* `--device` specifies the device used, default is `cpu` (support GPU with CUDA).

#### Example Input YAML File

This is an example input YAML file, which will be used as the default if no input file is provided:

```yaml
times:
    YYYY: 2000
    MM: 1
    DD: 1 # Start time
    days: 0
    hours: 0
    minutes: 0
    seconds: 1080 # Total run time
    dt_atmos: 1 # Time step

dims:
    global_npx: 201 # Number of meshes in x-dir, and number of control units is npx-1
    global_npy: 2 # Number of meshes in y-dir, and number of control units is npy-1
    npz: 300 # Number of layers in z-dir
    ng: 3 # Number of ghoost cells

fpses:
    px: 1 # Number of MPI partitions in x-dir
    py: 1 # Number of MPI partitions in y-dir
    
ics:
    ic_type: 21 # Case type.
                # 21: Robert gauss warm bubble

ios:
    do_output: True # ncfile output
    nc_parallel: True # Use parallel NetCDF
    out_fre: 10 # File output frequency; 10 means output every 10 time steps;
    do_diag: True # diag

dyns:
    k_split: 5
    n_split: 25 # Actual time step is dt/(k_split * n_split)
    rk: 3 # Time integration method
    dyn_core: eul    # eul_sw: Use shallow water core(1d or 2d)
                     # eul_sw_linear: Use shallow water core(1d or 2d)
                     # eul: Use normal eul core(3d only)
                     # eul_vic: Use vic dyn core
                     # eul_en: Use energy dyn core
                     # eul_en_vic: Use energy dyn core with vic
    lim: False # Use flux limiter
    lim_deg: 0 # limiter strength parameter
```
