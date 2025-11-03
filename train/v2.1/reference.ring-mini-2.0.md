---
base_model:
- inclusionAI/Ling-mini-base-2.0-20T
library_name: transformers
license: mit
pipeline_tag: text-generation
---

# Ring-mini-2.0

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*4QxcQrBlTiAAAAAAQXAAAAgAemJ7AQ/original" width="100"/>
<p>

<p align="center">🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a>
| &nbsp;&nbsp;🐙 <a href="https://zenmux.ai/inclusionai/ring-mini-2.0">Experience Now</a></p>

This model is presented in the paper [Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model](https://huggingface.co/papers/2510.18855).

Today, we officially release Ring-mini-2.0 — a high-performance inference-oriented MoE model deeply optimized based on the Ling 2.0 architecture. With only 16B total parameters and 1.4B activated parameters, it achieves comprehensive reasoning capabilities comparable to dense models below the 10B scale. It excels particularly in logical reasoning, code generation, and mathematical tasks, while supporting 128K long-context processing and 300+ tokens/s high-speed generation.

## Enhanced Reasoning: Joint Training with SFT + RLVR + RLHF
Built upon Ling-mini-2.0-base, Ring-mini-2.0 undergoes further training with Long-CoT SFT, more stable and continuous RLVR, and RLHF joint optimization, significantly improving the stability and generalization of complex reasoning. On multiple challenging benchmarks (LiveCodeBench, AIME 2025, GPQA, ARC-AGI-v1, etc.), it outperforms dense models below 10B and even rivals larger MoE models (e.g., gpt-oss-20B-medium) with comparable output lengths, particularly excelling in logical reasoning.

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_d2byvp/afts/img/O2YKQqkdEvAAAAAASzAAAAgADod9AQFr/original" width="1000"/>
              
<p>

## High Sparsity, High-Speed Generation
Inheriting the efficient MoE design of the Ling 2.0 series, Ring-mini-2.0 activates only 1.4B parameters and achieves performance equivalent to 7–8B dense models through architectural optimizations such as 1/32 expert activation ratio and MTP layers. Thanks to its low activation and high sparsity design, Ring-mini-2.0 delivers a throughput of 300+ tokens/s when deployed on H20. With Expert Dual Streaming inference optimization, this can be further boosted to 500+ tokens/s, significantly reducing inference costs for high-concurrency scenarios involving thinking models. Additionally, with YaRN extrapolation, it supports 128K long-context processing, achieving a relative speedup of up to 7x in long-output scenarios.
<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_d2byvp/afts/img/gjJKSpFVphEAAAAAgdAAAAgADod9AQFr/original" width="1000"/>
<p>

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_d2byvp/afts/img/o-vGQadCF_4AAAAAgLAAAAgADod9AQFr/original" width="1000"/>
<p>


## Model Downloads

<div align="center">

|     **Model**      | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :----------------: | :---------------: | :-------------------: | :----------------: | :----------: |
| Ring-mini-2.0 |       16.8B       |         1.4B         |        128K         |      [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ring-mini-2.0) <br>[🤖 Modelscope](https://modelscope.cn/models/inclusionAI/Ring-mini-2.0)|
</div>

## Quickstart

### 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ring-mini-2.0"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Ring, an assistant created by inclusionAI"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
model_inputs = tokenizer([text], return_tensors="pt", return_token_type_ids=False).to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=8192
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## License
This code repository is licensed under [the MIT License](https://huggingface.co/inclusionAI/Ring-mini-2.0/blob/main/LICENSE).

## Project Page
Access the demo and experience the model at: [https://zenmux.ai/inclusionai/ring-mini-2.0](https://zenmux.ai/inclusionai/ring-mini-2.0)

## Code
The full code repository for this model can be found on GitHub: [https://github.com/inclusionAI/Ring-V2](https://github.com/inclusionAI/Ring-V2)

## Citation
If you find our work helpful, feel free to give us a cite.
```bibtex
@article{lingteam2025every,
      title={Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model}, 
      author={Ling Team and Anqi Shen and Baihui Li and Bin Hu and Bin Jing and Cai Chen and Chao Huang and Chao Zhang and Chaokun Yang and Cheng Lin and Chengyao Wen and Congqi Li and Deng Zhao and Dingbo Yuan and Donghai You and Fagui Mao and Fanzhuang Meng and Feng Xu and Guojie Li and Guowei Wang and Hao Dai and Haonan Zheng and Hong Liu and Jia Guo and Jiaming Liu and Jian Liu and Jianhao Fu and Jiannan Shi and Jianwen Wang and Jianxin Lai and Jin Yang and Jun Mei and Jun Zhou and Junbo Zhao and Junping Zhao and Kuan Xu and Le Su and Lei Chen and Li Tang and Liang Jiang and Liangcheng Fu and Lianhao Xu and Linfeng Shi and Lisha Liao and Longfei Zheng and Meng Li and Mingchun Chen and Qi Zuo and Qiang Cheng and Qianggang Cao and Qitao Shi and Quanrui Guo and Senlin Zhu and Shaofei Wang and Shaomian Zheng and Shuaicheng Li and Shuwei Gu and Siba Chen and Tao Wu and Tao Zhang and Tianyu Zhang and Tianyu Zhou and Tiwei Bie and Tongkai Yang and Wang Hong and Wang Ren and Weihua Chen and Wenbo Yu and Wengang Zheng and Xiangchun Wang and Xiaodong Yan and Xiaopei Wan and Xin Zhao and Xinyu Kong and Xinyu Tang and Xudong Han and Xudong Wang and Xuemin Yang and Xueyu Hu and Yalin Zhang and Yan Sun and Yicheng Shan and Yilong Wang and Yingying Xu and Yongkang Liu and Yongzhen Guo and Yuanyuan Wang and Yuchen Yan and Yuefan Wang and Yuhong Guo and Zehuan Li and Zhankai Xu and Zhe Li and Zhenduo Zhang and Zhengke Gui and Zhenxuan Pan and Zhenyu Huang and Zhenzhong Lan and Zhiqiang Ding and Zhiqiang Zhang and Zhixun Li and Zhizhen Liu and Zihao Wang and Zujie Wen},
      year={2025},
      eprint={2510.18855},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
