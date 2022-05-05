# !/bin/bash
python3 train_pecnet_ethucy.py --dataset eth --gpu_num 0 && echo "eth Launched." &
P0=$!
python3 train_pecnet_ethucy.py --dataset hotel --gpu_num 1 && echo "hotel Launched." &
P1=$!
python3 train_pecnet_ethucy.py --dataset univ --gpu_num 2 && echo "univ Launched." &
P2=$!
python3 train_pecnet_ethucy.py --dataset zara1 --gpu_num 3 && echo "zara1 Launched." &
P3=$!
python3 train_pecnet_ethucy.py --dataset zara2 --gpu_num 4 && echo "zara2 Launched." &
P4=$!
wait $P0 $P1 $P2 $P3 $P4