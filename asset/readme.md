# Keye-VL-671B-A37B


Meet Keye-VL-671B-A37B — the most powerful multi-modal language model in the Keye series to date.

As one of the largest and most capable MLLMs currently in existence, Keye-VL-671B-A37B demonstrates top-tier and in some cases even leading performance in text understanding and generation, complex visual perception and reasoning, comprehensive video understanding, and Olympic-level mathematical reasoning.

#### Key Enhancements:

##### Pre-Training
* **Efficient Perception Building with Limited Compute**: We employ VisionEncoder from Keye-VL-1.5 and rigorously processed high-quality data to cost-effectively build the model’s core perceptual capabilities, ensuring strong visual understanding without excessive computational overhead.

* **Multi-Modal Data Curation**: We implement a automated data pipeline that performs strict filtering, re-sampling, and large-scale synthesis of structured VQA data, including OCR, charts, and tables. This end-to-end process significantly enhances the model’s perception quality and generalization.

* **Reasoning Sustainment via Synthetic CoT Data**: During the continual pretrain phase, we incorporate a diverse set of synthetically generated chain-of-thought (CoT) data. This ensures the model maintains its complex reasoning skills while progressing in perceptual pre-training.

##### Post-Training

* **Scaling law of Reasoning Data for SFT**: We experimentally validate that the mixed data (50B Instruct & Long-CoT data) improves model performance and training stability compared to the single model (30B Instruct data).
* **CoT Quality & Style Refinement**: We develop a data filtering process to remove redundant reflective chains, improving the model's reasoning and perception capabilities, with the in-house process outperforming GPT-4o.
* **High-Precision RL Verifier**: We train a dedicated verifier (Keye Verifier) to validate the model's reasoning  consistency and answer correctness, achieving significantly higher accuracy than other reward models and general LLMs, thereby enhancing our RL performance gains.

## Model Performance

![Performance Comparison](https://github.com/Kwai-Keye/Keye/blob/main/asset/radar.png)

![Performance on Public Benchmarks](https://github.com/Kwai-Keye/Keye/blob/main/asset/performance.png)

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

