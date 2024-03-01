# Karpathy's minBPE in Rust

[minbpe](https://github.com/karpathy/minbpe)

byte-pair encoder
made in order to practice rust i wouldnt imagine it to be following best practices atm as it's mostly identical to mingpts python code, will review reformat and try to make more rust like later.

#### todo: 

 - [x] BasicTokenizer
    - [x] train
    - [x] encode
    - [x] decode
    - [x] save
    - [x] load
    - [x] vocab type shud be vec u32?
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