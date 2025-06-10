#!/usr/bin/env python

# Purpose: run model
# Author: Chen Research Group (xichen.me@outlook.com)

#import warnings
#warnings.simplefilter('error', RuntimeWarning)

def run_model(fps, input: str, output: str) -> None:

    from lmarspy.model.ctrl import Ctrl

    ctrl = Ctrl(fps, input=input, output=output)
    
    for na in range(1, ctrl.num_atmos_calls+1):
        
        ctrl.step(na)

    return

def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(description='model dynamic core')
    parser.add_argument('--input', help='yaml filename of model input')
    parser.add_argument('--output', default='output', help='path to output (output.nc and output.log)')
    parser.add_argument('--backend', default='numpy', help='backend data type: numpy, torch')
    parser.add_argument('--device', default='cpu', help='the device used')
    args = parser.parse_args()

    from lmarspy.backend.field import Field
    Field.backend = args.backend
    Field.device = args.device

    from lmarspy.fps import Fps 
    fps = Fps(args.output)

    fps.init_model()
    
    run_model(fps, args.input, args.output)

    fps.end_model()

    return

if __name__ == "__main__":

    main()


