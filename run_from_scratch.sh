#!/bin/bash

rm -r workdir/
#rm -r ~/.cache/torch_extensions/
python main.py --config configs/numeric/cifar10_ncsnpp.py --mode train --workdir ./workdir


