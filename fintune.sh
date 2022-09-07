node_rank=$1

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5

nohup python -m torch.distributed.launch --nnodes=4 --master_addr=10.116.149.142 --node_rank=${node_rank}  --nproc_per_node=8   --master_port 23332  \
    --use_env main_finetune.py --output_dir output_dir/finetune_output_dir --log_dir  output_dir/finetune_output_dir \
    --finetune output_dir/dirty_no_postion/checkpoint-99.pth --batch_size 32 --min_lr 2e-4 \
    >./output_dir/finetune_output_dir/${node_rank}.log 2>&1 &