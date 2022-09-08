node_rank=$1

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5

nohup python -m torch.distributed.launch --nnodes=2 --master_addr=10.116.147.13 --node_rank=${node_rank}  --nproc_per_node=8   --master_port 23332  \
    --use_env main_finetune.py --output_dir /nlp_group/wuxing/gaochaochen/mae/output_dir/finetune_epochs800 \
    --log_dir  /nlp_group/wuxing/gaochaochen/mae/output_dir/finetune_epochs800 \
    --finetune  /nlp_group/wuxing/gaochaochen/mae/output_dir/pretrain_base_rdma_fp16/checkpoint-799.pth \
    --batch_size 64 \
    >/nlp_group/wuxing/gaochaochen/mae/output_dir/finetune_epochs800/${node_rank}.log 2>&1 &