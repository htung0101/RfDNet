#CUDA_VISIBLE_DEVICES=$1 python main.py --config configs/config_files/ISCNet_tdw_detection.yaml --mode train

CUDA_VISIBLE_DEVICES=$1 python main.py --config configs/config_files/ISCNet_tdw_detection_rgbd.yaml --mode train
#CUDA_VISIBLE_DEVICES=$1 python main.py --config configs/config_files/ISCNet_detection.yaml --mode train
#CUDA_VISIBLE_DEVICES=$1 python main.py --config configs/config_files/ISCNet_tdw_detection_rgbd.yaml --mode train


#CUDA_VISIBLE_DEVICES=$1 python main.py --config configs/config_files/ISCNet_tdw_completion.yaml --mode train

#CUDA_VISIBLE_DEVICES=$1 python main.py --config configs/config_files/ISCNet_completion.yaml --mode train