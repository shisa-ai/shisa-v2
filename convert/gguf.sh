cd ~/llama.cpp
# activate	
# install torch and transformers
pip install -r requirements.txt

# 7b takes about 1.5min
# bf16 converts to bf16/fp32 layers as appropriate
# requires folder
# 9 min for 70B
python convert-hf-to-gguf.py --outtype bf16 --outfile shisa-v1-llama3-70b.bf16.gguf shisa-v1-llama3-70b

# make - for quantize
make clean && make LLAMA_CUBLAS=1 -j96

# quantize - 3 min
time ./quantize shisa-v1-llama3-70b.bf16.gguf shisa-v1-llama3-70b.Q4_K-M.gguf Q4_K_M

# rest
time ./quantize shisa-v1-llama3-70b.bf16.gguf shisa-v1-llama3-70b.Q4_0.gguf Q4_0; time ./quantize shisa-v1-llama3-70b.bf16.gguf shisa-v1-llama3-70b.Q5_K-M.gguf Q5_K_M; ❯ time ./quantize shisa-v1-llama3-70b.bf16.gguf shisa-v1-llama3-70b.Q8_0.gguf Q8_0
