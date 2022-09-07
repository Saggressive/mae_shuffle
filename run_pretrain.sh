export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5

node_rank=$1

nohup python -m torch.distributed.launch --nnodes=8 --master_addr=10.116.149.142 --node_rank=${node_rank}  --nproc_per_node=8   --master_port 23333  \
    --use_env main_pretrain.py --output_dir output_dir/base_mae_no_postion --log_dir  output_dir/base_mae_no_postion \
    >output_dir/base_mae_no_postion/${node_rank}.log 2>&1 &
