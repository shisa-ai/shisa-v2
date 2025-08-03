# How to Train Qwen3-30B-A3B (MoE 30B) with Megatron-LM on AMD MI300X Using MegaBlocks

**Overview:** In this guide, we will walk through the process of fine-tuning **Qwen3-30B-A3B** – a 30.5B parameter Mixture-of-Experts (MoE) model – on an AMD Instinct MI300X 8-GPU node using **Megatron-LM (ROCm fork)** with **MegaBlocks** for efficient MoE training. We cover environment setup, data preparation, model configuration (including MoE specifics and parallelism settings), training execution, optimizer choices (standard Adam vs 8-bit vs Muon), precision options (BF16 vs FP8), and resource estimates (memory usage breakdown and expected training time).

**Assumptions:** We have access to an AMD Docker image with Megatron-LM (v0.10.0 ROCm development branch) and 8× MI300X GPUs (192GB HBM each). We will perform **full fine-tuning** (FFT) of Qwen3-30B-A3B for both **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)** stages, using sequence lengths 8K (SFT) and 4K (DPO) respectively. We assume that the base pretrained Qwen3-30B-A3B weights are available (or can be downloaded from Hugging Face) and that the fine-tuning datasets are prepared.

Let’s get started\!

## 1\. Environment Setup with AMD ROCm Megatron-LM

**Step 1.1: Pull and Launch the AMD Megatron-LM Docker Container.** AMD provides a pre-built Docker image (rocm/megatron-lm) optimized for MI300X GPUs with all necessary components (PyTorch, ROCm libs, Megatron-LM, etc.)[\[1\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=The%20Megatron,generation%20AI%20models%20more%20efficiently)[\[2\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=Ubuntu%2024). We will use the latest available image (e.g. v25.5) and run it with appropriate privileges:

\# Pull the AMD Megatron-LM Docker (Ubuntu 24.04 \+ Python3.12 variant)  
docker pull rocm/megatron-lm:v25.5\_py312

\# Launch the container with required devices and settings  
docker run \-it \--name megatron\_training\_env \\  
  \--device=/dev/dri \--device=/dev/kfd \--device=/dev/infiniband \\  
  \--network=host \--ipc=host \--group-add=video \\  
  \--cap-add=SYS\_PTRACE \--security-opt seccomp=unconfined \--privileged \\  
  \-v $HOME:$HOME \-v $HOME/.ssh:/root/.ssh \--shm-size 128G \\  
  rocm/megatron-lm:v25.5\_py312

This container already includes the **ROCm Megatron-LM** fork (a specialized version of Megatron-LM for AMD GPUs) pre-installed[\[3\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=The%20Docker%20container%20includes%20a,LM%2C%20including%20necessary%20training%20scripts). It contains optimized libraries: **Transformer Engine** (for mixed precision and FP8 support), **FlashAttention 3**, **hipBLASLt**, **Triton 3.3**, **RCCL** (NCCL for AMD), and supports advanced features like 3D parallelism (tensor, pipeline, sequence), fused kernels, etc[\[4\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=Transformer%20Engine)[\[5\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=). The AMD Megatron-LM is tuned to leverage MI300X hardware for large models (Meta Llama, DeepSeek, Mistral, etc.)[\[1\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=The%20Megatron,generation%20AI%20models%20more%20efficiently)[\[6\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=The%20following%20models%20are%20pre,AMD%20Instinct%20MI300X%20series%20accelerators), so Qwen3 (which is architecturally similar to a Llama-style MoE model) should be well-supported.

**Step 1.2: System Configuration (Recommended).** Before heavy training, ensure the host system is optimized for ROCm performance. For example, disable NUMA auto-balancing and fix GPU clocks to maximum for stability[\[7\]](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html#:~:text=1.%20System%20configuration)[\[8\]](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html#:~:text=,frequency). On the host (outside container):

\# Disable NUMA auto-balancing for better performance  
echo 0 | sudo tee /proc/sys/kernel/numa\_balancing

\# Set GPU clock frequency to high performance mode (e.g., 1900 MHz)  
rocm-smi \--setperfdeterminism 1900

These steps help eliminate system-related bottlenecks. (They are optional but recommended for optimal throughput.)

**Step 1.3: Install/Update Additional Packages in Container.** Once inside the container (bash prompt in the megatron\_training\_env), you might need to install some utilities. For example, ensure git is available to clone repositories, and install pip packages for any needed tools (like Jupyter or debug tools). The container likely has git and common build tools pre-installed, but if not:

apt update && apt install \-y git wget vim  \# install basics if needed

Also, the container should have ROCm 6.3+ and PyTorch 2.8 dev, which is sufficient for our needs[\[9\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=ROCm). Python is 3.12 (for this container tag). Verify that GPU devices are visible in PyTorch:

python \-c "import torch; print(torch.cuda.device\_count(), torch.cuda.get\_device\_properties(0))"

This should list 8 devices with something like "MI300X" in the name.

## 2\. Integrating MegaBlocks for Efficient MoE Training

**Step 2.1: Install the ROCm-compatible MegaBlocks library.** [**MegaBlocks**](https://github.com/databricks/megablocks) is an efficient MoE training library that reformulates MoE computations as block-sparse operations, eliminating token dropping and improving hardware utilization[\[10\]](https://github.com/databricks/megablocks#:~:text=MegaBlocks%20is%20a%20light,and%20standard%20MoE%20layers)[\[11\]](https://github.com/databricks/megablocks#:~:text=MegaBlocks%20dMoEs%20outperform%20MoEs%20trained,our%20paper%20for%20more%20details). MegaBlocks is integrated with Megatron-LM to support data, expert, and pipeline parallel MoE training[\[12\]](https://github.com/databricks/megablocks#:~:text=and%20standard%20MoE%20layers). However, the official MegaBlocks repo targets NVIDIA GPUs by default. AMD provides a **ROCm fork of MegaBlocks** that adds HIP support for AMD GPUs[\[13\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=efficiency%20and%20performance)[\[14\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=,installed).

To install MegaBlocks (ROCm version) inside the container:

\# Clone the ROCm fork of MegaBlocks and install it  
git clone https://github.com/ROCm/megablocks.git  
cd megablocks  
pip install .  \# this will build and install megablocks for ROCm

This should compile the block-sparse GPU kernels for HIP/ROCm and install the megablocks package. *(If any build issues arise, refer to the provided Dockerfile from the AMD blog for guidance[\[14\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=,installed)[\[15\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=,megablocks%2Fdocker%2FDockerfile). The Dockerfile may already include steps like installing pybind11 and other dependencies before building MegaBlocks.)*

**Step 2.2: Verify MegaBlocks Installation.** After installation, you can do a quick import test:

python \-c "import megablocks; print('MegaBlocks version:', megablocks.\_\_version\_\_)"

Also, ensure that MegaBlocks is linked with Megatron-LM. MegaBlocks provides extended layers and launch scripts for MoE integrated into Megatron. For instance, you should now have access to scripts under megablocks/exp and megablocks/megablocks modules, as well as modifications to Megatron to enable \--moe options.

**Step 2.3: Understand MegaBlocks \+ Megatron Integration.** With MegaBlocks, adding MoE to a transformer model is as simple as specifying a few **MoE hyperparameters** in the training script. MegaBlocks handles the rest by injecting MoE layers into the Megatron model definition. Key MoE arguments include[\[16\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=,to%20process%20each%20input%20token):

* \--moe-num-experts: **Number of experts** in each MoE layer (for Qwen3-30B-A3B, use 128).

* \--moe-top-k: **Top-K experts** to select per token (Qwen3 uses 8 experts per token, so moe-top-k=8).

* \--moe-capacity-factor: Capacity factor for expert load (controls expert buffer capacity; MegaBlocks often sets this to 1.0 meaning no token dropping[\[17\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=%23%20MoE%20hyperparameters.%20MOE_ARGUMENTS%3D%22%5C%20,k%3D1)).

* \--moe-loss-weight: Auxiliary load-balancing loss weight (helps ensure even expert utilization[\[18\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=,tokens%20an%20expert%20can%20process), e.g. 0.1).

In MegaBlocks, the costly all-to-all communications and padding from traditional MoE are optimized away by block-sparse kernels, yielding up to 40% speedup over standard MoE (like Tutel) and up to 2.4× speedup over dense models in some cases[\[11\]](https://github.com/databricks/megablocks#:~:text=MegaBlocks%20dMoEs%20outperform%20MoEs%20trained,our%20paper%20for%20more%20details). We will leverage this for Qwen3’s MoE layers.

## 3\. Preparing the Data in Megatron Format

**Step 3.1: Tokenizer and Vocabulary.** Qwen3 uses a custom tokenizer (likely similar to tiktoken or SentencePiece, supporting up to 128K context with YaRN). We need the tokenizer files for Qwen3 to preprocess data and tokenize on-the-fly. The easiest way is to use the Hugging Face model tokenizer:

* Get your HuggingFace access token (if needed for Qwen3 weights/tokenizer, which are Apache 2.0 licensed, so a token might not be required, but just in case).

* In the container, do:

* export HF\_TOKEN=\<your token\>

* Download Qwen3-30B-A3B’s tokenizer. The HuggingFace model card indicates the code is integrated into transformers\>=4.51.0 under the name 'qwen3\_moe'[\[19\]](https://huggingface.co/Qwen/Qwen3-30B-A3B#:~:text=Quickstart). If the AMD Megatron uses HuggingFaceTokenizer, we can set TOKENIZER\_MODEL="Qwen/Qwen3-30B-A3B" and it should download the files[\[20\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=Tokenizer)[\[21\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=The%20training%20script%20uses%20the,appropriate%20Hugging%20Face%20model%20path). For certainty, we can manually fetch the needed files:

* mkdir \-p tokenizer/qwen3-30b  
  wget \--header="Authorization: Bearer $HF\_TOKEN" \-O tokenizer/qwen3-30b/tokenizer.json \\  
      https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/main/tokenizer.json  
  \# (If Qwen uses sentencepiece, download sp.model and sp.vocab; if BPE, download merges.txt and vocab.json)

* According to Qwen’s GitHub, it might use a tokenizer.json (unified HuggingFace tokenizer file). Ensure you have the vocab and merges or the tokenizer config needed.

**Step 3.2: Format the Fine-Tuning Data.** Megatron-LM requires training data in a specific **binary format** for efficiency. Typically, you need to prepare a **JSONL** file where each line is a JSON object with a "text" field, then use Megatron’s preprocess\_data.py tool to create **indexed binary** files. For example, if you have a text dataset for SFT (supervised instructions and responses), gather them into one or multiple text files.

* **Combine text**: Ensure your SFT dataset is one document per line (or if multi-line dialogues, you might separate by an EOD token later).

* **Convert to JSONL**: Wrap each document in {"text": "...text content..."} and save to sft\_data.jsonl. Do the same for the DPO dataset (dpo\_data.jsonl).

**Step 3.3: Run Megatron Preprocessing.** Use the tools/preprocess\_data.py script from Megatron to create the binary dataset. We can run this inside the container (assuming we cloned Megatron-LM or it’s in /workspace/Megatron-LM in the container image):

cd /workspace/Megatron-LM

\# For SFT data  
python tools/preprocess\_data.py \\  
  \--input /path/to/sft\_data.jsonl \\  
  \--json-keys text \\  
  \--output-prefix /path/to/preprocessed/sft\_dataset \\  
  \--vocab-file /workspace/tokenizer/qwen3-30b/tokenizer.json \\  
  \--tokenizer-type HuggingFaceTokenizer \\  
  \--dataset-impl mmap \\  
  \--workers 8 \--append-eod

\# For DPO data  
python tools/preprocess\_data.py \\  
  \--input /path/to/dpo\_data.jsonl \\  
  \--json-keys text \\  
  \--output-prefix /path/to/preprocessed/dpo\_dataset \\  
  \--vocab-file /workspace/tokenizer/qwen3-30b/tokenizer.json \\  
  \--tokenizer-type HuggingFaceTokenizer \\  
  \--dataset-impl mmap \\  
  \--workers 8 \--append-eod

This will produce files like sft\_dataset\_text\_document.bin and .idx (and similarly for DPO). The \--append-eod flag adds an End-of-Document token between concatenated texts, which is useful if your data is a collection of separate prompts. We set tokenizer-type=HuggingFaceTokenizer and provided the Qwen tokenizer JSON (Megatron’s HuggingFaceTokenizer will load it accordingly[\[20\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=Tokenizer)).

**Step 3.4: Verify Data.** After preprocessing, verify that the binary files exist and possibly do a small test run with \--data-path pointing to them in a dry-run mode (like a few iterations) to ensure no tokenization errors. The AMD tutorial suggests using \--mock-data for testing connectivity, but here we have real data, so ensure the DATA\_PATH is set correctly in the script.

Now our dataset is ready in Megatron format.

## 4\. Configuring the Qwen3-30B-A3B Model in Megatron-LM

With data prepared, we configure the model and training run. AMD’s Megatron-LM repository provides example scripts in examples/ (e.g., train\_llama3.sh, train\_mixtral\_moe.sh, etc.)[\[22\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=Configuration)[\[23\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=Update%20the%20,as%20described%20in%20Run%20training). We can either modify one of these or create our own script for Qwen3. For clarity, we will outline key configuration variables and then present a unified training launch command.

**Step 4.1: Model Architecture Settings.** Qwen3-30B-A3B specifics (from the model card and blog):

* **Layers:** 48 transformer layers[\[24\]](https://qwenlm.github.io/blog/qwen3/#:~:text=Qwen3,4%20128%20%2F%208%20128K).

* **Hidden size & Heads:** Uses **Grouped Query Attention (GQA)** with 32 attention heads for Queries and 4 heads for Keys/Values[\[25\]](https://huggingface.co/Qwen/Qwen3-30B-A3B#:~:text=%2A%20Number%20of%20Paramaters%20%28Non,and%20131%2C072%20tokens%20with%20YaRN). This multi-query attention means the hidden dimension can be inferred: if each KV head has the same dimension as a Q head, then *hidden size* \= 32 \* head\_dim for Q. The head\_dim is not explicitly given in the card, but likely 128 (common in Qwen/Llama models). If head\_dim=128, hidden size \= 32*128 \=* *4096*\*. (However, 4096 might be on the small side for 30B parameters; Qwen3 might use a larger head\_dim or intermediate size. It’s possible the hidden is 5120 or 6144 with different head splitting, but for demonstration we’ll assume 4096 and rely on MoE to increase parameter count.)

* **Intermediate size:** For MoE layers, each expert’s feed-forward network (FFN) dimension. We will likely set this via \--ffn-hidden-size or similar. If hidden=4096 and using typical 4× expansion, dense intermediate would be \~16384. But since we have MoE, each expert could use a reduced intermediate. Qwen’s total activated parameters suggest each expert is smaller. We’ll tentatively use \--ffn-hidden-size 16384 (4× 4096\) unless more info indicates otherwise.

* **Experts:** 128 experts per MoE layer, with 8 selected per token (top-k=8)[\[25\]](https://huggingface.co/Qwen/Qwen3-30B-A3B#:~:text=%2A%20Number%20of%20Paramaters%20%28Non,and%20131%2C072%20tokens%20with%20YaRN).

* **MoE Layers:** It’s not explicitly stated if all 48 layers have MoE. The naming “A3B” implies 3.3B activated, which likely means every layer is MoE (for maximum capacity) but each expert is relatively small. We will assume **all layers are MoE layers** for the configuration (the MegaBlocks integration in Megatron will insert MoE at every layer if we set \--num-experts \> 0 for the model config).

* **Vocabulary size:** Qwen3’s vocab is large (supports 100+ languages and up to 128K context). The tokenizer file tokenizer.json will contain the actual vocab size (likely \~151k tokens as seen in the usage example[\[26\]](https://huggingface.co/Qwen/Qwen3-30B-A3B#:~:text=,except%20ValueError%3A%20index%20%3D%200) where a special token id 151668 appears, suggesting vocab \~151,669). We should set \--vocab-file and \--tokenizer-type accordingly and possibly \--vocab-size if needed (Megatron can usually infer from tokenizer, but it can be specified).

* **Max Position Embeddings:** 32,768 (native context)[\[27\]](https://huggingface.co/Qwen/Qwen3-30B-A3B#:~:text=,and%20131%2C072%20tokens%20with%20YaRN). We will set \--seq-length 8192 for SFT and adjust as needed for DPO (4096), but the model’s embedding matrix can handle up to 32k. We assume rotary or Alibi embeddings for extended context (Qwen uses a YaRN strategy for 131k with some trick, but we won’t cover beyond 32k here).

**Step 4.2: Parallelism Settings.** With 8 MI300X GPUs, we want to utilize them efficiently:

* **Data Parallel (DP):** By default, if we use no model parallel, Megatron will use data parallel across the 8 GPUs for batch distribution.

* **Tensor Parallel (TP):** Splits matrix computations across GPUs. Useful for large dense layers. Given hidden size 4096, TP isn’t strictly necessary for memory, but could increase compute throughput. We might set TP\_SIZE=2 or 4 to split each matrix across 2 or 4 GPUs, leaving the rest for expert parallel.

* **Pipeline Parallel (PP):** Splits layers among GPUs. With 8 GPUs and 48 layers, we could assign 6 layers per GPU for pipeline=8 (if not using them for other parallelism). However, introducing pipeline parallel increases sequence communication overhead. Since MoE itself distributes the heavy FFN load, we might prefer **data \+ expert \+ tensor parallel** without pipeline. So, we will set PP\_SIZE=1 (no pipeline splitting).

* **Expert Parallel (EP):** This is critical for MoE. We have 128 experts to distribute. A natural choice is **EP \= 8**, meaning we partition the experts across 8 GPUs[\[28\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=RECOMPUTE_NUM_LAYERS%3D4%20TEE_OUTPUT%3D1%20MBS%3D1%20GBS%3D16%20TP_SIZE%3D1,sh)[\[29\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=). Each GPU will host a subset of the experts for each MoE layer (16 experts per layer on each GPU, unique across GPUs). This way, when MoE layers are computed, each GPU handles its own experts, and an all-to-all communication gathers token outputs from the appropriate experts. MegaBlocks will handle this communication efficiently. We also set \--moe-expert-parallel flags accordingly via environment or CLI.

* We also ensure **Expert Tensor Parallel (ETP)** is set to 1 (meaning each expert’s weights are *not* further split across GPUs). If we were doing both TP and EP, an expert’s weight matrix could itself be split (ETP\>1), but to keep it simple, we’ll assume ETP=1 (like in AMD’s Mixtral example)[\[28\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=RECOMPUTE_NUM_LAYERS%3D4%20TEE_OUTPUT%3D1%20MBS%3D1%20GBS%3D16%20TP_SIZE%3D1,sh).

* **Sequence Parallel (SP):** AMD’s docs mention 3D parallelism including sequence parallel (SP) which partitions the sequence dimension to reduce activation memory. However, sequence parallel is typically used with ZeRO optimizer to partition gradients. We won’t explicitly use SP here unless the ROCm Megatron enables it by default with distributed optimizer. We will focus on DP/TP/EP.

**Step 4.3: Memory and Checkpointing Settings.** With 8192 sequence length, activation memory will be substantial. We enable **activation checkpointing** to trade compute for memory. Megatron allows \--checkpoint-activations (or through the env var AC=\<mode\> in AMD scripts). We will use **AC=full** (full checkpoint of all layers) for maximum memory savings at the cost of recomputation[\[30\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=). This means during backprop, intermediate activations will be recomputed instead of stored, drastically cutting activation memory usage.

We also set a reasonable **micro-batch size** per GPU to balance utilization and memory. MI300X has 192GB, which is very high, so we can push batch size. Suppose we target a **global batch size** of 128 sequences for SFT (with seq len 8192). If DP=1 (no replication because EP is using all GPUs), effectively each GPU might handle 16 sequences (since 128/8=16) per step. That could be done as micro-batch \= 2 and gradient accumulation steps \= 8 to reach 16, or micro-batch=4, grad acc=4, etc. We’ll start with simpler: MBS=2 (micro-batch per GPU) and let the script accumulate to reach GBS=128.

For DPO (seq len 4096), we can use a larger batch (maybe global 256 or 512\) since shorter sequences use less memory. We can adjust these later.

**Step 4.4: Initialize from Pretrained Weights.** Since we are fine-tuning, we need to load Qwen3-30B-A3B’s base weights. The ROCm Megatron might not have a direct one-click way to load HF weights, but we can do one of two things: \- **Option A:** Convert the HuggingFace model to Megatron checkpoint format. Megatron-LM provides scripts for converting certain models (like GPT-2, GPT-NeoX, etc.). For a custom architecture like Qwen3, a conversion script might need to be written. Essentially, we would instantiate the HF model in PyTorch, then map its weights to a Megatron model state dict (with MoE layers). This can be complex, but possible. Due to time, we might instead: \- **Option B:** Use the **HuggingFace load within Megatron**. AMD’s fork allows using the HF tokenizer and they mention some models (Llama) needing license. If a conversion is too involved, one might simply initialize training from scratch or from a smaller checkpoint. *However*, since Qwen3-30B is a large model, starting from scratch for fine-tuning is not ideal.

Given the complexity, let’s assume we have converted and saved a Megatron-compatible checkpoint of Qwen3-30B-A3B (perhaps by using a combination of the HF model and Megatron’s checkpoint.py). We would then use \--load /path/to/qwen30b\_checkpoint in the training command to load it.

*(If conversion is not done, one could initially run a few hundred steps from scratch just to validate pipeline, but for actual fine-tuning, obtaining the pretrained weights in Megatron format is necessary to achieve good results.)*

**Step 4.5: Assemble the Training Command.** We can either set environment variables and call the example script, or call pretrain\_gpt.py directly with arguments. The AMD container’s examples/ scripts expect env vars for convenience. We will illustrate using an environment variable approach similar to AMD’s docs:

For **Single-Node Training** on 8 GPUs, with MoE (EP=8), BF16 precision:

\# Navigate to Megatron-LM directory  
cd /workspace/Megatron-LM

\# Export environment variables for configuration:  
export TEE\_OUTPUT=1                   \# to see aggregated logs in stdout  
export MBS=2                          \# micro batch size per GPU  
export GBS=128                        \# global batch size (total across all GPUs)  
export TP\_SIZE=1                      \# tensor parallelism (1 \= no TP, adjust if desired)  
export PP\_SIZE=1                      \# pipeline parallelism (1 \= no PP)  
export EP\_SIZE=8                      \# expert parallelism (using 8 GPUs for experts)  
export ETP\_SIZE=1                     \# expert tensor parallel (1 \= each expert on one GPU)  
export AC=full                        \# activation checkpointing (full)  
export PR=bf16                        \# precision (bf16 for training)  
export SEQ\_LENGTH=8192                \# sequence length (max seq for this run)  
export DATA\_PATH="/path/to/preprocessed/sft\_dataset\_text\_document"  \# without file extension  
export TOKENIZER\_MODEL="Qwen/Qwen3-30B-A3B"  \# huggingface model name for tokenizer, if needed  
export SAVE\_PATH="/path/to/output/checkpoints"  
export LOAD\_PATH="/path/to/pretrained/qwen30b\_megatron\_checkpoint"  \# if available

\# Launch the training script for Qwen (we can adapt llama script as Qwen3 is similar)  
bash examples/llama/train\_llama3.sh  \# using Llama3 script as a template

We would need to **edit train\_llama3.sh** to reflect Qwen’s model dimensions and MoE. For example, open examples/llama/train\_llama3.sh and adjust:

* Set NUM\_LAYERS=48

* Set HIDDEN\_SIZE=\<appropriate hidden size\> (if we assume 4096 or a different value like 5120 – we need to confirm Qwen’s hidden dim. If using 4096, also adjust FFN hidden size accordingly).

* Set NUM\_ATTENTION\_HEADS=32 and perhaps an option for KV heads. If AMD’s Megatron supports GQA, it might have an argument like \--num-kv-heads 4 or similar (NVIDIA Megatron had \--num-query-groups for MQA). Check if train\_llama3.sh mentions KV head sharing; in the performance data we saw “FP8-KV” which hints at that. If not, we might approximate by treating it like standard multi-head (32 heads) and accept the slight architecture mismatch (it will still function).

* Set MoE flags: e.g. \--num-experts 128, \--top-k 8 in the script’s GPT\_ARGS or similar. The AMD Mixtral script likely has something like:

* GPT\_ARGS="... \--num-experts $EP\_SIZE \--moe-expert-parallel-size $EP\_SIZE \--moe-top-k 8 \--moe-loss-weight 0.1 \--capacity-factor 1.0 ..."

* If not automatically set by EP env, we add them manually.

* Set dataset path (DATA\_PATH) to our SFT data for the first fine-tuning stage.

* Set saving and logging options (like \--save-interval, \--log-interval as needed).

Once configured, running the script will launch the training on all 8 GPUs (the docker run we used already gave the container all 8 GPUs and set \--ipc host etc., which are needed for NCCL communication).

**Important:** Ensure NCCL network interface is set inside the container. The script will use NCCL\_SOCKET\_IFNAME environment variable. If not set, set it to your network interface (e.g., export NCCL\_SOCKET\_IFNAME=eth0 or whatever interface connects multi-node, though for single node it’s not critical). AMD’s documentation shows how to detect and set it[\[31\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=ip%20a)[\[32\]](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html#:~:text=import%20os%20import%20subprocess).

**Step 4.6: Monitoring Training with Weights & Biases (Optional).** We want to report training metrics (loss, throughput, etc.) to Weights & Biases. Megatron-LM does not natively integrate with W\&B, but we can still use it: \- Install wandb in the container: pip install wandb. \- Before running training, login or set API key: wandb login (and follow prompts) or export WANDB\_API\_KEY=.... \- You can then run the training script under a wandb agent or simply use wandb.init() in a custom training loop. Since we are using Megatron’s scripts, a simple way is to run the training inside a wandb context by prefixing the command:

wandb run \-- python pretrain\_gpt.py \[all args\]

However, an easier method is to use the TEE\_OUTPUT logs: since TEE\_OUTPUT=1 writes combined log to a file (usually ./output.log), you could tail that file and push metrics to W\&B manually. A more direct approach is to modify Megatron’s training loop to call wandb.log() each iteration for loss and throughput. This requires editing the Megatron code slightly (e.g., in Megatron-LM/megatron/training.py, inside the training iteration loop). \- If that’s too involved, at minimum you can capture final metrics and manually log them to W\&B after training.

For now, consider W\&B optional and proceed with training. You will still get periodic printouts of training loss, learning rate, etc., in the console or output.log[\[33\]\[34\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=iteration%20%20%20%20,iterations%3A%20%20%200). You can monitor GPU utilization with rocm-smi and system metrics as usual.

## 5\. Optimizer Choices and Memory Trade-offs

Given the model size and training constraints, choosing the right optimizer is crucial for both memory usage and performance. We will discuss three optimizer options: **Baseline AdamW (FP32 states)**, **8-bit AdamW**, and **Muon** optimizer.

### 5.1 Baseline: AdamW with FP32 States

By default, Megatron-LM uses AdamW for pretraining/fine-tuning. This maintains two state tensors (momentum and variance) in FP32 for each weight parameter, which doubles memory usage compared to the model weights. For Qwen3-30B-A3B: \- **Model weights:** \~30.5B \* 2 bytes (BF16) ≈ **61 GB** total (if stored on one GPU). In our case with EP=8, each GPU stores 1/8 of the MoE weights and a full copy of dense weights. Roughly, each GPU might hold \~21 GB of weights in BF16 (see breakdown below). \- **Optimizer states (FP32):** Two states per weight in 4 bytes each \= 30.5B \* 8 bytes \= **244 GB** if on one GPU. But Megatron’s **distributed optimizer** will *shard* these states across GPUs to save memory[\[35\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=,SP%20%2B%20CP). With 8 GPUs, ideally each holds \~1/8 of the optimizer states (\~30.5 GB each). In practice, Megatron’s “distributed optimizer” (similar to ZeRO Stage-1) will partition the optimizer state for data-parallel groups. Because we have expert parallel with essentially no data parallel replication, we expect the optimizer state for the dense weights (which are replicated across GPUs) to be sharded, while each GPU solely updates its own expert weights (no replication to shard there). Net result: **Optimizer memory per GPU** might be on the order of **30–40 GB** in FP32 Adam.

Given MI300X has 192 GB per GPU, even FP32 Adam is feasible. However, it will consume a large chunk of memory and reduce the headroom for activations and batch size. It’s also slower in terms of memory bandwidth and reduction operations.

### 5.2 8-bit Optimizer (bitsandbytes or PyTorch 2.5)

Using 8-bit precision for optimizer states can **cut memory by 4×** without significantly impacting model quality[\[36\]](https://huggingface.co/papers?q=Muon%20optimizer#:~:text=Daily%20Papers%20,bit%20optimizer%20states.%20To). There are two main avenues: \- **bitsandbytes**: A popular library providing an 8-bit Adam optimizer and int8 quantization for LLMs. It is supported on AMD ROCm as of ROCm 6.2+[\[37\]](https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html#:~:text=bitsandbytes%20is%20a%20Python%20wrapper,refer%20to%20the%20ROCm%20documentation). AMD even provides a pre-built wheel (e.g., bitsandbytes-0.44.1 for ROCm)[\[38\]](https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html#:~:text=,directly%20install%20from%20wheel%20package). We can install it via:

pip install \--no-deps \--force-reinstall \\  
  https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release\_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux\_2\_24\_x86\_64.whl

This installs bitsandbytes with ROCm support. We can then use bitsandbytes.optim.AdamW8bit as drop-in for AdamW. \- **PyTorch built-in 8-bit (Torch-**

AO

): New versions of PyTorch are expected to integrate 8-bit optimizers (possibly under torch.optim or torch.ao.optim). If such an optimizer is available (for example, torch.optim.AdamW(..., foreach=True, capturable=True, optim\_dtype=torch.uint8) – hypothetical), it could be used instead of bitsandbytes. As of now, the easiest route is bitsandbytes.

To use bitsandbytes in Megatron, one approach is to modify the training script to initialize the optimizer with bitsandbytes. If Megatron’s trainer code detects optimizer\_type \== "adamw\_8bit", it could call bitsandbytes. If not, you can monkey-patch after the model is built:

import bitsandbytes as bnb  
optimizer \= bnb.optim.AdamW8bit(model.parameters(), lr=\<learning\_rate\>, betas=(0.9,0.95), weight\_decay=1e-5)

This might require custom integration. Alternatively, you could run Megatron with ZERO\_OPTIMIZATION off and manually manage optimizer outside. But given time, it might be easier to rely on AMD’s distributed optimizer for sharding and just trust memory is enough.

**Memory Savings:** With 8-bit Adam, each state is one byte instead of four, so the two states per weight use 2 bytes total instead of 8\. That’s a 4× reduction in optimizer memory. E.g., instead of \~30 GB per GPU, it might be \~7.5 GB. This is a **significant savings**, allowing bigger batches or just more slack.

**Speed:** 8-bit operations are slightly more CPU-bound due to quantization overhead, but on GPUs with high memory bandwidth like MI300X, the difference is minor. AMD’s blog notes up to 4× less memory with negligible performance loss using bitsandbytes 8-bit optimizers[\[39\]](https://huggingface.co/docs/bitsandbytes/en/optimizers#:~:text=8,bit%20optimizers).

**Recommendation:** If comfortable, use the 8-bit optimizer for fine-tuning to utilize the 192GB more for activations and throughput. Bitsandbytes is proven on ROCm[\[40\]](https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html#:~:text=implement%20these%20on%20AMD%20GPUs,using%20ROCm), so it’s a reliable choice.

### 5.3 Muon Optimizer

**Muon** is an alternative optimizer specifically targeting the **fully-connected layers** of neural networks (i.e., the large weight matrices in hidden layers)[\[41\]](https://github.com/KellerJordan/Muon#:~:text=Muon%3A%20An%20optimizer%20for%20the,hidden%20layers%20of%20neural%20networks)[\[42\]](https://github.com/KellerJordan/Muon#:~:text=Muon%20is%20an%20optimizer%20for,should%20be%20used%20as%20follows). It was introduced to speed up training and potentially improve convergence for large models by using a variant of heavy-ball momentum (and Nesterov acceleration) on the main weight matrices, while still using AdamW on biases, embeddings, and layer norms[\[43\]](https://github.com/KellerJordan/Muon#:~:text=%23%20optimizer%20%3D%20torch.optim.AdamW%28model.parameters%28%29%2C%20lr%3D3e,01).

**How to use Muon:** After installing it (pip install git+https://github.com/KellerJordan/Muon), you can set up the optimizer as:

from muon import MuonWithAuxAdam  
hidden\_weights \= \[p for p in model.parameters() if p.ndim \>= 2\]  \# matrices  
hidden\_biases\_and\_other \= \[p for p in model.parameters() if p.ndim \< 2\]  
optimizer \= MuonWithAuxAdam(\[  
    {"params": hidden\_weights, "use\_muon": True, "lr": \<lr\_large\>, "weight\_decay": \<wd\_large\>},  
    {"params": hidden\_biases\_and\_other, "use\_muon": False, "lr": \<lr\>, "betas": (0.9,0.95), "weight\_decay": \<wd\>}  
\])

Muon uses a different learning rate for the large matrices (often higher) and theoretically can lead to faster training in terms of steps (some reports claim improved loss per step).

**Memory Impact:** Muon’s main advantage is not memory quantization but possibly *not using second-moment statistics for the large params*. It functions more like SGD+Momentum for those weights. This means it stores only one momentum tensor for those (instead of two like Adam). If that’s the case, it can nearly halve the memory for those layers’ optimizer states. Meanwhile, biases and other small params still use Adam (two states, but those are a tiny fraction of total params). So Muon could cut optimizer memory roughly in half relative to full AdamW (e.g., \~15–20 GB per GPU instead of 30 GB). Muon does **not** compress to 8-bit, so it’s not as memory-light as bitsandbytes, but it reduces state count.

**Considerations:** Muon is newer and may require hyperparameter tuning (especially the lr for hidden weights). It’s been shown to maintain stability even with large batch sizes[\[44\]](https://github.com/KellerJordan/Muon#:~:text=,training%20with%20large%20batch%20size). If you are adventurous, you could try Muon for faster convergence; otherwise, an 8-bit AdamW is more straightforward for ensuring comparable results to baseline.

### 5.4 Summary – Optimizer Choice

For most users, we recommend using **AdamW with 8-bit states** as a good balance of familiarity and memory efficiency. If maximum stability is needed and memory is plentiful, standard AdamW (with distributed sharding) in BF16/FP32 is fine. If pushing for cutting-edge speed, Muon could be experimented with, possibly in combination with 8-bit momentum (though Muon itself doesn’t have an 8-bit implementation to our knowledge).

We will proceed assuming **AdamW (distributed) with 8-bit states**.

**Tip:** If using bitsandbytes’s 8-bit Adam, set optimizer.zero\_grad(set\_to\_none=True) to avoid accumulating gradients, and be mindful that gradient clipping might need adaptation (bitsandbytes has some nuances there).

## 6\. Precision: BF16 vs FP8 on MI300X

The MI300X supports training in BF16 by default (as do all modern GPUs). It also supports **FP8** training through the integrated **Transformer Engine (TE)** in Megatron[\[45\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=Transformer%20Engine)[\[46\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=). FP8 can further boost throughput and reduce memory, but requires careful loss scaling and is only recommended if the software (Transformer Engine) handles it robustly (which it generally does for pretraining large models).

**BF16 (Brain Float 16):** We use BF16 for safer training if FP8 is uncertain. BF16 has the same range as FP32 for the exponent, so it rarely underflows/overflows in practice, and MI300X achieves excellent performance with BF16 tensor cores. All our above memory estimates assumed BF16 for model weights and activations. BF16 is a **good default**.

**FP8 (E5M2 or E4M3):** Transformer Engine dynamically manages FP8 with scaling per tensor. AMD’s container includes Transformer Engine 1.13 which presumably supports FP8 GEMMs on MI300X[\[4\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=Transformer%20Engine). The **Key options** allow enabling FP8 globally by setting environment TE\_FP8=1 and PR=fp8 (precision)[\[47\]\[46\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=). When enabled, certain matrix multiplications (attention projections, MLP weights) will be done in FP8. Weights may be stored as FP8 with a margin or cast on the fly (NVIDIA’s TE keeps a FP16 master and an FP8 copy for forward).

**Can MI300X do FP8?** Yes – AMD specifically lists FP8 training support in their docs and performance results (e.g., Llama-70B FP8 runs)[\[48\]](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html#tabs-a8deaeb413-item-21cea50186-tab#:~:text=Llama%203.1%2070B%20%28amd%2FLlama,7). So MI300X *can* do FP8. Using FP8 will: \- Speed up training (more FLOPS utilized, faster GEMMs). \- Reduce memory for activations by 2× (since activations can be FP8 in caches) and possibly for some optimizer steps if adapted. \- *Risk:* Slightly more difficult convergence. But since Qwen3 has already been trained and we are fine-tuning, FP8 should likely be okay, especially if combined with gradual FP8 introduction or maintaining critical layers in BF16 if needed.

**Recommendation:** Try **BF16 first** for a few epochs to ensure stable training. Then, if you need faster training, you can attempt FP8. The AMD example uses FP8 even for large Llama models and shows good scaling[\[48\]](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html#tabs-a8deaeb413-item-21cea50186-tab#:~:text=Llama%203.1%2070B%20%28amd%2FLlama,7). If enabling FP8, keep an eye on loss curves and ensure no divergence. Use \--fp8-e4m3 or \--fp8-hybrid if options exist to choose FP8 format. Transformer Engine will handle scaling.

## 7\. Training Execution and Monitoring

With everything configured, run the training script (as shown in Step 4.5). It will output logs like:

iteration    50/xxxxx | consumed tokens:  xx | elapsed time per iteration: xx ms | learning rate: xx | global batch: xx | lm loss: x.xxxxxx | load balance loss: x.xxxxx | ...

For example, in AMD’s GPT-2 MoE test, they saw loss decreasing and reported the load balancing loss (MoE auxiliary) around 0.099, indicating experts were used evenly[\[34\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=iteration%20%20%20%20,iterations%3A%20%20%200)[\[49\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=From%20the%20output%2C%20you%20can,indicating%20effective%20learning). We should see two losses if MoE is on: the primary LM loss and the MoE load balance loss. Ensure the load balance loss is not zero (if it is exactly zero, perhaps the \--moe-loss-weight is not set or experts are imbalanced). Aim for it to hover around the weight we set (0.1), meaning the gating is actively balancing.

**During training**, monitor: \- **Throughput:** You can derive tokens/sec from the log. E.g., if global batch \=128 and seq=8192, that’s 1,048,576 tokens per iteration. If iteration time is 800 ms, that’s \~1.31 million tokens/sec across the cluster, \~163k tokens/sec/GPU on average. Use this to estimate total time. \- **Memory usage:** Use rocm-smi in another terminal to see GPU Memory usage. It should ideally be below \~180GB to have some safety margin. If it’s near 192GB, consider reducing batch or sequence (or verify if some memory is not properly freed). \- **Gradients:** If any NaN or inf appears, you might need to lower LR or use gradient clipping (\--clip-grad 1.0 is common). Also check if any expert is not utilized (gating issues).

**Switching from SFT to DPO:** After SFT (say 300M token training, which might be a few epochs depending on data size), you will then fine-tune with DPO at 4K context. You can reuse the same setup but change SEQ\_LENGTH=4096, point DATA\_PATH to DPO dataset, and possibly adjust batch up (e.g., GBS=256 or 512 since sequence is half length, memory allows more). Also consider resetting some optimizer states if needed or using the final SFT model as initialization for DPO. Since DPO is a preference tuning (like RLHF but via direct loss), ensure prompts are formatted appropriately and maybe reduce LR a bit because DPO might be sensitive.

## 8\. Memory Usage Breakdown (Estimate)

Finally, let’s break down memory usage per GPU on the MI300X node for this training, using approximate numbers:

* **Model Weights:** Each GPU holds:

* Full copy of **dense weights** (attention layers, embed, etc.) – roughly \~7.8B parameters (embedding \+ QKV and output matrices across 48 layers, by earlier estimate) \~ **15.6 GB** in BF16[\[3\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=The%20Docker%20container%20includes%20a,LM%2C%20including%20necessary%20training%20scripts).

* A shard of **MoE weights** – with EP=8, each GPU has \~1/8 of the MoE parameters. Total MoE params \~22.7B (if all layers MoE). So per GPU \~2.84B params \~ **5.7 GB** in BF16.

* **Total weights per GPU ≈ 21.3 GB** in BF16.

* **Optimizer States:** (assuming 8-bit Adam, distributed)

* For dense weights (replicated across GPUs): optimizer states can be sharded, so each GPU may hold \~1/8 of those states. Dense \~7.8B params globally; two states would be 7.8B×2 \= 15.6B elements. Sharded 8 ways \= 1.95B elements per GPU. At 1 byte each (8-bit), that’s \~1.95 GB.

* For MoE weights (unique to each GPU): each GPU solely updates its 2.84B params. Two states per param \= 5.68B elements *1 byte \=* *5.68 GB*\*.

* **Total optimizer memory per GPU ≈ 1.95 \+ 5.68 \= 7.6 GB**.

* *(If using FP32 Adam: it would be \~4× larger: dense part \~7.8 GB, MoE part \~22.7 GB, total \~30.5 GB per GPU, but again sharded dense means \~26 GB per GPU in that case. Muon would be \~15–20 GB per GPU by halving states for large params.)*

* **Activations:** This is batch-dependent. For SFT (seq 8192, global batch 128, micro 2 per GPU):

* Per GPU per step: 2 sequences \* 8192 tokens \= 16,384 tokens processed concurrently. Each token produces activations of size \[hidden\_size\] for forward, and gradients of same for backward.

* Activation memory roughly \= (micro\_batch\_size \* seq\_length \* hidden\_size \* bytes) \* (\# of layers that are not checkpointed).

* With **full checkpointing**, we only hold activations for one layer at a time (plus a few for recompute). So estimate one layer’s activation: 2 \* 8192 \* 4096 \* 2 bytes ≈ 2 \* 8192 \* 4096 \* 2 ≈ **134,217,728 bytes (\~128 MB)** per layer per GPU in forward. That gets recomputed in backward, so not stored for all layers.

* Additionally, at backward pass, one layer’s gradients might be held similarly. So it might peak slightly above that per layer. But since only a couple layers’ activations are in memory at once, this is manageable.

* There are also **attention key/value caches** for 8192 sequence. With multi-query attn (4 KV heads), the KV cache per layer is seq\_len \* kv\_heads \* head\_dim (if head\_dim=128, kv\_heads=4, that’s 8192*4*128 \= 4,194,304 elements per layer per GPU, \~8MB in BF16). Over 48 layers \~384MB if not freed. FlashAttention v3 might compute on the fly to avoid large caches.

* **Total activations**: Rough estimate \~ a few GB (say 2–6 GB) per GPU, depending on implementation. This is relatively small due to checkpointing. Without checkpointing, it would easily exceed 30 GB.

* **Other Memory:**

* **Gradients**: Megatron will accumulate gradients for micro-batches. These gradient tensors are same size as weights (BF16 or FP32). If distributed optimizer, part of gradient memory is also sharded. We might budget \~5–10 GB for gradients.

* **Temporary buffers**: for all-reduce, activation recomputation, workspace for GEMMs, etc. Possibly a couple GB.

* **FlashAttention**: if using, it may allocate some workspace (\~a few hundred MB).

Summing the above per GPU: \- Weights: \~21 GB \- Optimizer: \~7.6 GB (8-bit) or \~30 GB (FP32) \- Activations: \~4 GB (with checkpointing) \- Gradients \+ overhead: \~8 GB \- **Total \~40 GB (with 8-bit opt) or \~63 GB (with full FP32 opt)** per GPU.

These fit well under 192 GB. In fact, it leaves a large margin that you can use to increase batch size or add pipeline parallel if needed. This margin also accounts for any fragmentation or extra loads.

**Memory Chart:**

| Memory Component | Per GPU (approx) | Notes |
| :---- | :---- | :---- |
| **Model Weights (BF16)** | \~21 GB | (Dense \~15.6 GB \+ MoE \~5.7 GB)[\[3\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=The%20Docker%20container%20includes%20a,LM%2C%20including%20necessary%20training%20scripts) |
| **Optimizer States** | \~7.6 GB (8-bit) \<br\> \~30 GB (FP32) | (Distributed across GPUs)[\[37\]](https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html#:~:text=bitsandbytes%20is%20a%20Python%20wrapper,refer%20to%20the%20ROCm%20documentation)[\[38\]](https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html#:~:text=,directly%20install%20from%20wheel%20package) |
| **Activations (Checkpointed)** | \~4 GB | (Sequence 8192, micro-batch 2\) |
| **Gradients & Buffers** | \~8 GB | (Accumulated grads, all-reduce bufs, etc.) |
| **Total** | \~40–45 GB (with 8-bit opt) \<br\> \~60–65 GB (with FP32 opt) | Out of 192 GB available (plenty of headroom) |

*These are estimates. Actual usage observed via rocm-smi may vary.* The comfortable headroom means we could potentially double the batch (to global 256\) and still fit \~80–90 GB usage. Indeed, MI300X’s abundant memory is a big advantage here.

## 9\. Expected Training Time (GPU-Hours)

We now estimate the training duration for the given token counts:

* **SFT stage:** 300M tokens.

* **DPO stage:** 100M tokens.

* **Total:** 400M tokens.

Our throughput depends on batch size and hardware. Let’s use a conservative estimate from similar setups: \- In AMD’s performance tests, **Llama-70B** on 8×MI300X achieved \~426 TFLOPs at 8192 seq (with FSDP)[\[50\]](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html#tabs-a8deaeb413-item-21cea50186-tab#:~:text=Llama%203,79), which might correspond to roughly 0.5M to 1M tokens/sec cluster-wide. Qwen3-30B is about half the size (active) so likely faster. \- Suppose we achieve **500k tokens/second** across 8 GPUs (this could be on the lower side if batch is smaller). At 500k/sec: \- 400,000,000 tokens / 500,000 \= **800 seconds** \= \~13.3 minutes. (That seems *too* fast; perhaps our token/sec guess is high). \- Let’s be more cautious: perhaps **100k tokens/sec** cluster-wide (which is likely an underestimate): \- 400M / 100k \= 4000 seconds \= \~1.11 hours. \- A more realistic middle ground: **200k–300k tokens/sec**: \- 400M / 250k \= 1600 seconds \= \~0.45 hours (27 minutes).

The variance is huge because it depends on how well we utilize the GPUs (batch size, FP8 usage, etc.). In practice, fine-tuning often is **I/O or dataloader-limited** if not careful, or slowed by small batches.

Given we have large batch and everything loaded in GPU memory, we should be GPU-limited. It’s safe to assume on the order of **hours, not days** for 400M tokens on this powerful node. To be safe, let’s estimate around **8–12 hours wall-clock time** for the full 400M tokens, which corresponds to **64–96 GPU-hours** (8 GPUs). This accounts for some inefficiencies, variations in sequence length between SFT and DPO, and possibly multiple epochs if needed.

For example, if each iteration (processing GBS=128 \* 8192 tokens \= \~1.05M tokens) takes \~2 seconds (which is a modest \~0.5M tokens/sec), 300M tokens would be \~285 iterations, which is 570 seconds (\~9.5 minutes) for SFT. DPO 100M tokens maybe \~3 minutes. Combined \~12.5 minutes. That is the optimistic scenario. If it’s 10× slower (5M tokens/sec cluster, i.e. 0.1M tokens/sec), it’s \~125 minutes (\~2 hours). If even slower, perhaps up to 6–8 hours in worst case.

Given the MI300X’s capability and our high-memory, high-throughput configuration (especially if FP8 is enabled), we lean towards the training being completed in well under a day. **A reasonable guess: \~100 GPU-hours in total**, e.g., \~12.5 hours on 8 GPUs.

This is remarkably fast compared to training from scratch, thanks to the high-end hardware and efficient software stack (Megatron \+ MegaBlocks \+ FlashAttention \+ FP8).

**Note:** Always monitor actual throughput. After a few hundred iterations, you can project remaining time. If using W\&B, you can log tokens/sec or iterations/sec to get a live estimate.

## 10\. Additional Tips and Conclusion

* **Validation/Evaluation:** We skipped validation in this how-to (since user just wants to train and evaluate at end). It’s wise to periodically evaluate the model on a small validation set (if available) to ensure it’s converging and not overfitting or diverging.

* **Saving Checkpoints:** The scripts usually save checkpoints at intervals. Ensure \--save points to a directory with enough space (each checkpoint will be \~60GB in size if it includes all experts – maybe more if not sharded). You might save only final model or use incremental saving carefully due to size.

* **Resuming:** If you need to resume training, use \--load with the last checkpoint. Megatron-LM will handle resuming iteration count and optimizer states.

* **Multi-Node Scalability:** Our focus was single-node (8 GPUs). If in future you use multiple MI300X nodes, you can expand NNODES and follow AMD’s multi-node launch instructions (using Slurm or mpirun). Ensure the network (InfiniBand or RoCE) is configured and that NCCL\_SOCKET\_IFNAME is set to the right interface on all nodes[\[32\]](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html#:~:text=import%20os%20import%20subprocess)[\[51\]](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html#:~:text=os.environ,active_interface). MegaBlocks and Megatron support multi-node MoE with expert parallel – they will increase EP size if you have more GPUs.

* **DeepSpeed vs Megatron:** The user mentioned familiarity with DeepSpeed \+ HF Transformers. Here, we effectively bypass DeepSpeed by using Megatron’s native distributed optimizer and parallelism, which is sufficient. DeepSpeed MoE (with Tutel) currently only supports up to ZeRO-2 and can have overhead[\[52\]](https://github.com/databricks/megablocks#:~:text=MegaBlocks%20is%20a%20light,and%20standard%20MoE). By using MegaBlocks, we avoid token dropping and get better speed. If one ever wants to integrate DeepSpeed (for ZeRO-3, etc.), it would require disabling Megatron’s own distributed features – not recommended for this case.

By following this guide, you should be able to fine-tune Qwen3-30B-A3B on your AMD MI300X system efficiently. The combination of **ROCm Megatron-LM** and **MegaBlocks** provides a powerful solution for MoE models, taking full advantage of MI300X’s ample memory and compute. Good luck with your training, and enjoy the lightning-fast fine-tuning of a 30B MoE model\!

**Sources:**

* AMD ROCm Documentation – *Training a model with Megatron-LM*[\[1\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=The%20Megatron,generation%20AI%20models%20more%20efficiently)[\[3\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=The%20Docker%20container%20includes%20a,LM%2C%20including%20necessary%20training%20scripts)[\[46\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=)

* AMD ROCm Blog – *Efficient MoE training on AMD GPUs (MegaBlocks)*[\[13\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=efficiency%20and%20performance)[\[16\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=,to%20process%20each%20input%20token)

* Qwen3 Model Card (HuggingFace) – *Model Overview and Specs*[\[25\]](https://huggingface.co/Qwen/Qwen3-30B-A3B#:~:text=%2A%20Number%20of%20Paramaters%20%28Non,and%20131%2C072%20tokens%20with%20YaRN)[\[24\]](https://qwenlm.github.io/blog/qwen3/#:~:text=Qwen3,4%20128%20%2F%208%20128K)

* AMD ROCm Blog – *8-bit Optimizer on AMD GPUs*[\[37\]](https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html#:~:text=bitsandbytes%20is%20a%20Python%20wrapper,refer%20to%20the%20ROCm%20documentation)[\[38\]](https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html#:~:text=,directly%20install%20from%20wheel%20package)

* Keller et al. – *Muon Optimizer Usage*[\[42\]](https://github.com/KellerJordan/Muon#:~:text=Muon%20is%20an%20optimizer%20for,should%20be%20used%20as%20follows)[\[53\]](https://github.com/KellerJordan/Muon#:~:text=hidden_weights%20%3D%20,4%2C%20betas%3D%280.9%2C%200.95%29%2C%20weight_decay%3D0.01%29%2C)

---

[\[1\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=The%20Megatron,generation%20AI%20models%20more%20efficiently) [\[2\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=Ubuntu%2024) [\[3\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=The%20Docker%20container%20includes%20a,LM%2C%20including%20necessary%20training%20scripts) [\[4\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=Transformer%20Engine) [\[5\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=) [\[6\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=The%20following%20models%20are%20pre,AMD%20Instinct%20MI300X%20series%20accelerators) [\[9\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=ROCm) [\[20\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=Tokenizer) [\[21\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=The%20training%20script%20uses%20the,appropriate%20Hugging%20Face%20model%20path) [\[22\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=Configuration) [\[23\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=Update%20the%20,as%20described%20in%20Run%20training) [\[28\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=RECOMPUTE_NUM_LAYERS%3D4%20TEE_OUTPUT%3D1%20MBS%3D1%20GBS%3D16%20TP_SIZE%3D1,sh) [\[29\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=) [\[30\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=) [\[31\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=ip%20a) [\[35\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=,SP%20%2B%20CP) [\[45\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=Transformer%20Engine) [\[46\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=) [\[47\]](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html#:~:text=) Training a model with Megatron-LM for ROCm — ROCm Documentation

[https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html)

[\[7\]](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html#:~:text=1.%20System%20configuration) [\[8\]](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html#:~:text=,frequency) [\[32\]](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html#:~:text=import%20os%20import%20subprocess) [\[51\]](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html#:~:text=os.environ,active_interface) Pretraining with Megatron-LM — Tutorials for AI developers 4.0

[https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup\_tutorial.html](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html)

[\[10\]](https://github.com/databricks/megablocks#:~:text=MegaBlocks%20is%20a%20light,and%20standard%20MoE%20layers) [\[11\]](https://github.com/databricks/megablocks#:~:text=MegaBlocks%20dMoEs%20outperform%20MoEs%20trained,our%20paper%20for%20more%20details) [\[12\]](https://github.com/databricks/megablocks#:~:text=and%20standard%20MoE%20layers) [\[52\]](https://github.com/databricks/megablocks#:~:text=MegaBlocks%20is%20a%20light,and%20standard%20MoE) GitHub \- databricks/megablocks

[https://github.com/databricks/megablocks](https://github.com/databricks/megablocks)

[\[13\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=efficiency%20and%20performance) [\[14\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=,installed) [\[15\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=,megablocks%2Fdocker%2FDockerfile) [\[16\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=,to%20process%20each%20input%20token) [\[17\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=%23%20MoE%20hyperparameters.%20MOE_ARGUMENTS%3D%22%5C%20,k%3D1) [\[18\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=,tokens%20an%20expert%20can%20process) [\[33\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=iteration%20%20%20%20,iterations%3A%20%20%200) [\[34\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=iteration%20%20%20%20,iterations%3A%20%20%200) [\[49\]](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html#:~:text=From%20the%20output%2C%20you%20can,indicating%20effective%20learning) Efficient MoE training on AMD ROCm: How-to use Megablocks on AMD GPUs — ROCm Blogs

[https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html](https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html)

[\[19\]](https://huggingface.co/Qwen/Qwen3-30B-A3B#:~:text=Quickstart) [\[25\]](https://huggingface.co/Qwen/Qwen3-30B-A3B#:~:text=%2A%20Number%20of%20Paramaters%20%28Non,and%20131%2C072%20tokens%20with%20YaRN) [\[26\]](https://huggingface.co/Qwen/Qwen3-30B-A3B#:~:text=,except%20ValueError%3A%20index%20%3D%200) [\[27\]](https://huggingface.co/Qwen/Qwen3-30B-A3B#:~:text=,and%20131%2C072%20tokens%20with%20YaRN) Qwen/Qwen3-30B-A3B · Hugging Face

[https://huggingface.co/Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)

[\[24\]](https://qwenlm.github.io/blog/qwen3/#:~:text=Qwen3,4%20128%20%2F%208%20128K) Qwen3: Think Deeper, Act Faster | Qwen

[https://qwenlm.github.io/blog/qwen3/](https://qwenlm.github.io/blog/qwen3/)

[\[36\]](https://huggingface.co/papers?q=Muon%20optimizer#:~:text=Daily%20Papers%20,bit%20optimizer%20states.%20To) Daily Papers \- Hugging Face

[https://huggingface.co/papers?q=Muon%20optimizer](https://huggingface.co/papers?q=Muon%20optimizer)

[\[37\]](https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html#:~:text=bitsandbytes%20is%20a%20Python%20wrapper,refer%20to%20the%20ROCm%20documentation) [\[38\]](https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html#:~:text=,directly%20install%20from%20wheel%20package) [\[40\]](https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html#:~:text=implement%20these%20on%20AMD%20GPUs,using%20ROCm) Quantized 8-bit LLM training and inference using bitsandbytes on AMD GPUs — ROCm Blogs

[https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html](https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html)

[\[39\]](https://huggingface.co/docs/bitsandbytes/en/optimizers#:~:text=8,bit%20optimizers) 8-bit optimizers \- Hugging Face

[https://huggingface.co/docs/bitsandbytes/en/optimizers](https://huggingface.co/docs/bitsandbytes/en/optimizers)

[\[41\]](https://github.com/KellerJordan/Muon#:~:text=Muon%3A%20An%20optimizer%20for%20the,hidden%20layers%20of%20neural%20networks) [\[42\]](https://github.com/KellerJordan/Muon#:~:text=Muon%20is%20an%20optimizer%20for,should%20be%20used%20as%20follows) [\[43\]](https://github.com/KellerJordan/Muon#:~:text=%23%20optimizer%20%3D%20torch.optim.AdamW%28model.parameters%28%29%2C%20lr%3D3e,01) [\[44\]](https://github.com/KellerJordan/Muon#:~:text=,training%20with%20large%20batch%20size) [\[53\]](https://github.com/KellerJordan/Muon#:~:text=hidden_weights%20%3D%20,4%2C%20betas%3D%280.9%2C%200.95%29%2C%20weight_decay%3D0.01%29%2C) GitHub \- KellerJordan/Muon: Muon is an optimizer for hidden layers in neural networks

[https://github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon)

[\[48\]](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html#tabs-a8deaeb413-item-21cea50186-tab#:~:text=Llama%203.1%2070B%20%28amd%2FLlama,7) [\[50\]](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html#tabs-a8deaeb413-item-21cea50186-tab#:~:text=Llama%203,79) Performance Results with AMD ROCm™ Software

[https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html)
