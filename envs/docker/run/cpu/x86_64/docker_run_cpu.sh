#!/bin/bash

docker run --rm -it \
    -v ${PWD}/../../../../../../lmarspy:/lmarspy \
    -v ${PWD}:/work --workdir /work \
    -v ${PWD}/../../examples:/examples \
    lmarspy/cpu/x86-64:base \
    bash /examples/run_robert_gauss_2d.sh
