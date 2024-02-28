# Karpathy's minBPE in Rust

[minbpe](https://github.com/karpathy/minbpe)

byte-pair encoder
made in order to practice rust i wouldnt imagine it to be following best practices atm as it's mostly identical to mingpts python code, will review reformat and try to make more rust like later.

#### todo: 

 - [ ] BasicTokenizer
    - [x] train
    - [x] encode
    - [x] decode
    - [ ] save
    - [ ] load
    - [ ] vocab type shud be u32;2?
 - [x] REPL <- (next)
    - [ ] correct prints/whitespaces
 - [ ] CLI
 - [ ] Validate results <- (next)
 - [ ] Set-up Tests <- (next)
    - [ ] self
    - [ ] vs minbpe
    - [ ] vs tiktoken
 - [ ] RegexTokenizer
 - [ ] GPT4Tokenizer
 - [ ] Tests + Compare
 - [ ] Structs Traits:?
 - [ ] Review, Reorg, rustify
 - [ ] pyo3 python lib?