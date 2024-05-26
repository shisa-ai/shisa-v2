The `get-tokenizer-efficiency.py` script will pull an chunk (`{lang}_part_00004`, chosen arbitrarily) from [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) and use it to calculate English and Japanese tokenizer efficiency.

Prior results are cached in the json files, md has human readable tables.

You may want to cross-reference with some more tokenizer tests I did from last year (using the same CulturaX dataset for calculation, so directly comparable): https://github.com/AUGMXNT/shisa/wiki/Tokenizer-Efficiency
