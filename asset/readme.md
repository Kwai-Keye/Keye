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

### Environment Setup

```shell
docker run -it --gpus all lmsysorg/sglang:v0.5.2
# make sure each node use the following commands to install the custom SGLang branch
git clone -b keye-dpsk-infer-fp8-release https://github.com/Kwai-Keye/sglang.git sglang
pip install -e sglang/python[all]
```

### Two-Node H800x8 Deployment

#### Prerequisites:

- Model: Kwai-Keye/Keye-VL-671B-A37B
- Node 1 IP: 192.168.1.100 (MASTER_NODE_IP)
- Node 2 IP: 192.168.1.101 (WORKER_NODE_IP)

#### Node 1 (Master - rank 0):

```shell
MODEL_PATH=/path/to/Keye-VL-671B-A37B
DIST_INIT_ADDR="MASTER_NODE_IP:29500"   # e.g. 192.168.1.100:29500
PORT=30000                              # listening port on each node
python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host 0.0.0.0 \
    --port $PORT \
    --tp-size 16 \
    --nnodes 2 \
    --node-rank 0 \
    --dist-init-addr $DIST_INIT_ADDR \
    --trust-remote-code \
    --mm-attention-backend fa3 \
    --attention-backend fa3 \
    --disable-radix-cache \
    --mem-fraction-static 0.8 \
    --cuda-graph-max-bs 64 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}'
```

#### Node 2 (Worker - rank 1):

```shell
MODEL_PATH=/path/to/Keye-VL-671B-A37B
DIST_INIT_ADDR="MASTER_NODE_IP:29500"   # e.g. 192.168.1.100:29500
PORT=30000                              # listening port on each node
python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host 0.0.0.0 \
    --port $PORT \
    --tp-size 16 \
    --nnodes 2 \
    --node-rank 1 \
    --dist-init-addr $DIST_INIT_ADDR \
    --trust-remote-code \
    --mm-attention-backend fa3 \
    --attention-backend fa3 \
    --disable-radix-cache \
    --mem-fraction-static 0.8 \
    --cuda-graph-max-bs 64 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}'
```

For more deployment details, please refer to the [Keye-VL-671B-A37B Deployment Tutorial](https://github.com/Kwai-Keye/sglang/blob/keye-dpsk-infer-fp8-release/scripts/deploy_keye_deepseek/DEPLOY_TUTORIAL.md).

### Client Usage

```python
import json
import requests

BASE_URL = "http://MASTER_NODE_IP:30000"

def generate(messages):
    payload = {
        "model": "",
        "messages": messages,
        "n": 1,
        "temperature": 0.0,
        "max_tokens": 256,
        "top_k": 1,
        "ignore_eos": False,
        "skip_special_tokens": True,
    }
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=1800,
    )
    resp.raise_for_status()
    return resp.json()

# Example: image + text
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png"},
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

result = generate(messages)
print(result["choices"][0]["message"]["content"])
```

