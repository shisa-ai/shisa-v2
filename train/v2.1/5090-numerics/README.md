
## 1 x MI300 Runs
Conclusions:
- DeepSpeed is necessary for optimal training - FP32 parameter upcasting leads to much better loss!
- The speed gains from disabling gradient checkpointing are significant - it's worth lowering batch size
- For AdamW, the torchao 8bit is just as good as the full precision version

| Run      | Optimizer       | DSZ3 |  MBS |  GAS | GC  |  tok/s |   mem |   Time |   Loss |
| :------- | :-------------- | :--: | ---: | ---: | :-- | -----: | ----: | -----: | -----: |
| [h3fhc1o0](https://wandb.ai/augmxnt/5090-numerics/runs/h3fhc1o0) | adam_torch_8bit |  No  |    8 |    8 | yes | 26.6K  | 18.18 |   N/A  |   N/A  |
| [bjsf8tlt](https://wandb.ai/augmxnt/5090-numerics/runs/bjsf8tlt) | adam_torch_8bit |  No  |   64 |    1 | yes | 23.1K  | 88.43 | 34m58s | 0.9877 |
| [2gswdtiq](https://wandb.ai/augmxnt/5090-numerics/runs/2gswdtiq) | adam_torch      |  No  |   64 |    1 | yes | 25.3K  | 84.82 |   N/A  |   N/A  |
| [hap954mv](https://wandb.ai/augmxnt/5090-numerics/runs/hap954mv) | adam_torch_8bit | Yes  |   64 |    1 | yes | 22.1K  | 95.04 | 37m17s | 0.90117 |
| [6q7gfxcs](https://wandb.ai/augmxnt/5090-numerics/runs/6q7gfxcs) | adam_torch_8bit | Yes  |    8 |    8 | no  | 29.4K  | 89.78 | 19m01s | 0.9033 |

- Canceled run confirming MBS=64 GAS=1 ~= MBS=8 GAS=8
- Canceled run confirming adam_torch == adam_torch_8bit
