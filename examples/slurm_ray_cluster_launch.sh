#!/bin/bash
#SBATCH -A your_account
#SBATCH --job-name=ray_cluster_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=224
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --partition=interactive
#SBATCH --exclusive

###############################################################################
# Multi-node Ray cluster job                                                   #
#  - One node runs the Ray head                                               #
#  - All nodes (incl. head) expose all 8 GPUs to Ray                          #
#  - Head node chosen as the first host in $SLURM_NODELIST                    #
###############################################################################

set -euo pipefail

# The script executes on the first node (the job launcher node).  Determine
# hostnames in this allocation:
SCRIPT_NODE=$(hostname -s)  # Node where this batch script is running
NODES=( $(scontrol show hostnames $SLURM_NODELIST) )

# Reorder such that SCRIPT_NODE is treated as head; preserve order for workers
HEAD_NODE=$SCRIPT_NODE
WORKER_NODES=()
for h in "${NODES[@]}"; do
  if [[ "$h" != "$SCRIPT_NODE" ]]; then
    WORKER_NODES+=("$h")
  fi
done

# Obtain IP of the chosen head node explicitly via srun
HEAD_IP=$(srun --nodes=1 --ntasks=1 --nodelist=${HEAD_NODE} bash -lc "hostname -I | awk '{print \$1}'")
export RAY_HEAD_IP=$HEAD_IP
export RAY_ADDRESS=ray://$HEAD_IP:10001
export RAY_TMPDIR="/tmp/${USER}/ray"

# Start Ray head on head node **with full resources** so it can execute
# workloads itself.  --block keeps the process in the foreground so the Slurm
# job stays alive for the full wall-time.  No separate worker on the head to
# avoid port collisions.
srun --nodes=1 --ntasks=1 --nodelist=${HEAD_NODE} \
  ray start --head \
    --temp-dir=$RAY_TMPDIR \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --num-gpus=8 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --include-dashboard=true \
    --node-ip-address=$HEAD_IP \
    --metrics-export-port=8080 \
    --ray-client-server-port=10001 \
    --block &

# Give head a few seconds to start before connecting workers
sleep 10

# Loop over remaining nodes, start workers via srun so they run remotely
for WORK_NODE in "${WORKER_NODES[@]}"; do
  echo "Launching Ray worker on ${WORK_NODE}"
  srun --nodes=1 --ntasks=1 --nodelist=${WORK_NODE} \
       ray start --address=$HEAD_IP:6379 \
                 --num-cpus=$SLURM_CPUS_PER_TASK \
                 --num-gpus=8 \
                 --block &
done

# Wait for background worker processes (remote nodes) until allocation ends
wait 