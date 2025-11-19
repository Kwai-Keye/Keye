# Keye-VL-671B-A37B


Meet Keye-VL-671B-A37B — the most powerful multi-modal language model in the Keye series to date.

As one of the largest and most capable MLLMs currently in existence, Keye-VL 671B demonstrates achieved top-tier (T1) — and in some cases even leading — performance in text understanding and generation, complex visual perception and reasoning, comprehensive video understanding, and Olympic-level mathematical reasoning.

#### Key Enhancements:

##### Pre-Training
* **Efficient Perception Building with Limited Compute**: We employ VisionEncoder from KeyeVL1.5 and rigorously processed high-quality data to cost-effectively build the model’s core perceptual capabilities, ensuring strong visual understanding without excessive computational overhead.

* **Multi-Modal Data Curation**: We implement a automated data pipeline that performs strict filtering, re-sampling, and large-scale synthesis of structured content—including OCR, charts, and tables—augmented by VQA data. This end-to-end process significantly enhances the model’s perception quality and generalization.

* **Reasoning Sustainment via Synthetic CoT Data**: During the Continue-Pretrain phase, we incorporate a diverse set of synthetically generated chain-of-thought (CoT) data. This ensures the model maintains its complex reasoning skills while progressing in perceptual pre-training.

##### Post-Training

* **Scaling law of Reasoning Data for SFT**: We experimentally validate that the mixed data (50B Instruct & Long-CoT data) improves model performance and training stability compared to the single model (30B Instruct data).
* **CoT Quality & Style Refinement**: We develop a data filtering process to remove redundant reflective chains, improving the model's reasoning and perception capabilities, with the in-house process outperforming GPT-4o.
* **High-Precision RL Verifier**: We train a dedicated verifier (Keye Verifier) to validate the model's reasoning  consistency and answer correctness, achieving significantly higher accuracy than other reward models and general LLMs, thereby enhancing our RL performance gains.

## Model Performance

![Performance Comparison](https://github.com/Kwai-Keye/Keye/blob/main/asset/radar.png)

|                       | Benchmarks     | Seed1.5-VL thinking | dots.vlm1 | Qwen3-VL-235B-A22B thinking | Keye-VL-1.5-671B-A37B |
| --------------------- | -------------- | :-----------------: | :-------: | :-------------------------: | :-------------------: |
| STEM/Reasoning        | MMMU_VAL       |        77.9         |   80.11   |            80.6             |       **83.78**       |
|                       | MMMU_Pro       |        67.6         |   70.11   |            69.3             |       **72.49**       |
|                       | MathVision     |        68.7         |   69.64   |          **74.6**           |         69.11         |
|                       | MathVista      |        85.6         |   85.0    |            85.8             |       **86.2**        |
|                       | OlympiadBench  |        65.0         |     -     |              -              |       **74.92**       |
|                       | VisuLogic      |        35.0         |   32.2    |            34.4             |       **35.4**        |
| General VQA           | RealWorldQA    |        78.4         |   79.08   |            81.3             |       **86.54**       |
|                       | MMStar         |        77.8         |   76.67   |            78.7             |       **86.67**       |
|                       | MMBench-en     |        89.9         |   89.32   |            90.6             |       **95.74**       |
|                       | MMbench-cn     |        89.1         |   88.24   |              -              |       **94.27**       |
|                       | MMVP           |        69.3         |   72.0    |              -              |       **88.0**        |
|                       | V*             |        89.0         |     -     |              -              |       **90.05**       |
|                       | HallusionBench |        60.3         |   64.83   |            66.7             |       **72.3**        |
| Video                 | VideoMME       |        77.9         |     -     |          **79.0**           |       **79.0**        |
|                       | LongVideoBench |        74.0         |     -     |         65.2 (fp8)          |       **79.0**        |
|                       | MMVU           |        70.1         |     -     |         78.4 (fp8)          |       **86.6**        |
|                       | TempCompass    |      **83.7**       |     -     |         81.03 (fp8)         |         77.75         |
| Text Recog./Doc/chart | TextVQA        |      **81.8**       |     -     |              -              |         76.21         |
|                       | DocVQA_VAL     |      **96.9**       |   96.52   |            96.5             |         95.39         |
|                       | ChartQA_TEST   |      **89.1**       |   87.68   |              -              |         86.68         |
|                       | InfoVQA        |      **91.2**       |     -     |            89.5             |         86.93         |
|                       | CharXiv (RQ)   |        60.2         |   64.4    |            66.1             |       **79.4**        |
|                       | CharXiv (DQ)   |        92.6         |   92.1    |              -              |       **94.5**        |
|                       | AI2D_TEST      |        87.3         |   88.37   |            89.2             |       **91.19**       |
| Pure Text             | AIME2025       |          -          |   85.83   |          **89.7**           |         83.3          |
|                       | GPQA           |          -          | **72.78** |              -              |         71.21         |

## Quickstart

### Using SGLang to Deploy Keye-VL-671B

1. make sure below `run.sh` is executeable on all nodes.

```shell
#!/bin/bash


if [ $# -ne 6 ]; then
    echo "Usage: ./run.sh <model_path> <dist_init_addr> <port> <node_rank> <stdout_log_path> <stderr_log_path>"
    exit 1
fi

model_path=$1
dist_init_addr=$2
port=$3
node_rank=$4
stdout_log_path=$5
stderr_log_path=$6

# load conda, path should be changed to the actual path on your cluster
# source /path/to/conda.sh
# conda activate /path/to/your_conda_env_with_sglang

NCCL_SOCKET_IFNAME=bond0 \
NCCL_IB_DISABLE=0 \
NCCL_DEBUG=INFO \
NCCL_IB_GID_INDEX=3 \
NCCL_NET_PLUGIN=none \
NCCL_IB_ECE_ENABLE=0 \
NVSHMEM_IB_ENABLE_IBGDA=0 \
NCCL_NVLS_ENABLE=0 \
NCCL_IB_TIMEOUT=22 \
NVSHMEM_IB_GID_INDEX=3 \
MC_TE_METRIC=True \
GLOO_SOCKET_IFNAME=bond0 \
MC_ENABLE_DEST_DEVICE_AFFINITY=True \
SGL_ENABLE_JIT_DEEPGEMM=True \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m sglang.launch_server \
    --model-path=$model_path \
    --host=0.0.0.0 \
    --port=$port \
    --tp-size=32 \
    --dp-size=32 \
    --ep-size=32 \
    --nnodes=4 \
    --dist-init-addr=$dist_init_addr \
    --trust-remote-code \
    --disable-radix-cache \
    --mem-fraction-static=0.7 \
    --mm-attention-backend=fa3 \
    --attention-backend=fa3 \
    --chunked-prefill-size=262144 \
    --moe-a2a-backend=deepep \
    --enable-two-batch-overlap \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --moe-dense-tp-size=1 \
    --watchdog-timeout=1000000 \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 32}' \
    --node-rank=$node_rank \
    1>$stdout_log_path \
    2>$stderr_log_path
```

2. Start model deployment on multiply nodes with `multi_node_deploy.sh`:

```shell
#!/bin/bash


# Configuration
MODEL_PATH="/path/to/model"
DIST_INIT_ADDR="192.168.1.100:29500"
BASE_PORT=30000
LOG_DIR="/path/to/logs"
mkdir -p $LOG_DIR

# Node IPs
NODES=("192.168.1.100" "192.168.1.101" "192.168.1.102" "192.168.1.103")

# Launch on each node
for i in "${!NODES[@]}"; do
    node_ip="${NODES[$i]}"
    node_rank=$i
    port=$((BASE_PORT + i))

    echo "Launching on node $node_rank (IP: $node_ip)"

    ssh $node_ip "cd /path/to/sglang && \
        ./scripts/deploy_keye_deepseek/run.sh \
        $MODEL_PATH \
        $DIST_INIT_ADDR \
        $port \
        $node_rank \
        $LOG_DIR/node${node_rank}_stdout.log \
        $LOG_DIR/node${node_rank}_stderr.log" &
done

wait
echo "All nodes launched"
```

More deployment information is available at  https://github.com/Kwai-Keye/sglang/blob/keye-dpsk-infer-fp8-release/scripts/deploy_keye_deepseek/DEPLOY_TUTORIAL.md 