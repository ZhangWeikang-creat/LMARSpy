#!/bin/bash

docker run --gpus all --rm -it --runtime=nvidia \
    -v ${PWD}/../../../../../../lmarspy:/lmarspy \
    -v ${PWD}:/work --workdir /work \
    -v ${PWD}/../../examples:/examples \
    lmarspy/cuda/x86-64:base \
    bash /examples/run_robert_gauss_2d.sh
