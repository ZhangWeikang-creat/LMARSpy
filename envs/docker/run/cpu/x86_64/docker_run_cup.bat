@echo off

set WORKDIR=%cd%

docker run --gpus all --rm -it --runtime=nvidia ^
    -v %WORKDIR%\..\..\..\..\..\..\lmarspy:/lmarspy ^
    -v %WORKDIR%:/work --workdir /work ^
    -v %WORKDIR%\..\..\examples:/examples ^
    lmarspy/cpu/x86-64:base ^
    bash \examples\run_robert_gauss_2d.sh

