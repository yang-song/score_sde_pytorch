#!/bin/bash

#DSM
#sigma=0.2
#export LOSS_SIGMA=$sigma
#rm -r ./workdir_${sigma}
#python main.py --config configs/vp/cifar10_ncsnpp.py --mode train --workdir ./workdir_${sigma}


export DEBUG=""
export DEBUG="$DEBUG full_backward_trajectories"
#export DEBUG="$DEBUG full_forward_trajectories"
#export DEBUG="$DEBUG terminal_forward_samples"
export DEBUG="$DEBUG importance_sampler_weighting"
export DEBUG="$DEBUG training_loss"
export DEBUG="$DEBUG backward_score_evaluations"
export DEBUG="$DEBUG score_function_scaling"
                    



#SSM
rm -r ./workdir_SSM 
#rm -r ./loss_vs_t_dump && mkdir loss_vs_t_dump


mv debug_data.h5 debug_data_last.h5

if [[ "x$1" == "xhardreset" ]]; then
	  rm -r ~/.cache/torch_extensions/
fi

python main.py --config configs/vp/cifar10_ncsnpp.py --mode train --workdir ./workdir_SSM
#python main.py --config configs/vp/cifar10_ncsnpp.py --mode eval --workdir ./workdir_SSM_radermacher



