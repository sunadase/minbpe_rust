use core::fmt;
use std::borrow::{Borrow, BorrowMut};
use std::cell::RefCell;
use std::char::REPLACEMENT_CHARACTER;
use std::env::args;
use std::ffi::OsString;
use std::fmt::format;
use std::num::ParseIntError;
use std::{fs, path};
use std::io::{self, stdin, stdout, BufRead, Error, Stdin, Write};
use std::path::PathBuf;
use std::rc::Rc;
use std::str::Bytes;
use std::sync::Arc;
use std::{collections::HashMap, io::Read, path::Path};
use std::cmp::{min_by_key, Eq, Ord, Reverse};
use std::collections::{BinaryHeap};
use std::hash::Hash;

mod utils;
use utils::most_frequent;

enum Ordering {
    Ascending,
    Descending
}

fn frequent_pair<T>(array: &[T], k: usize, ordering:Ordering) -> Vec<(usize, (&T, &T))>
where
    T: Hash + Eq + Ord,
{
    match ordering {
        Ordering::Ascending => {
            let mut map: HashMap<(&T, &T), usize> = HashMap::new();
            for pair in array.iter().zip(array.iter().skip(1)) {
                *map.entry(pair).or_default() += 1;
            }
        
            let mut heap = BinaryHeap::with_capacity(k + 1);
            for (x, count) in map.into_iter() {
                heap.push((count, x));
                if heap.len() > k {
                    heap.pop();
                }
            }
            return heap.into_sorted_vec().into_iter().collect();   
        },
        Ordering::Descending => {
            let mut map: HashMap<(&T, &T), usize> = HashMap::new();
            for pair in array.iter().zip(array.iter().skip(1)) {
                *map.entry(pair).or_default() += 1;
            }
        
            let mut heap = BinaryHeap::with_capacity(k + 1);
            for (x, count) in map.into_iter() {
                heap.push(Reverse((count, x)));
                if heap.len() > k {
                    heap.pop();
                }
            }
            return heap.into_sorted_vec().into_iter().map(|r| r.0).collect();   
        }
    }
}


fn merge(ids: &Vec<u32>, pair:&(u32, u32), idx:&u32) -> Vec<u32>{
    let mut newids: Vec<u32> = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if ids[i] == pair.0 && i < ids.len() - 1 && ids[i+1] == pair.1{
            newids.push(idx.clone());
            i += 2;
        } else {
          newids.push(ids[i].clone());
          i +=1;  
        }
    }
    return newids;
}

#[derive(Debug)]
struct BasicTokenizer {
    trained: bool,
    vocab_size: u32,
    num_merges: u32,

    merges: HashMap<(u32,u32),u32>,
    vocab: HashMap<u32, Vec<u32>>
    //vocab: HashMap<u32, String>
}

impl BasicTokenizer {
    fn train(text:&String, vocab_size:u32, verbose:Option<bool>) -> Self{
        let verbos = verbose.unwrap_or(false);

        assert!(vocab_size>=256, "vocab_size has to be larger than 256");
        let num_merges = vocab_size - 256;

        let mut merges: HashMap<(u32, u32), u32> = HashMap::new();
        let mut vocab: HashMap<u32, Vec<u32>> = HashMap::new();

        let mut ids:Vec<u32> = text.chars().map(u32::from).collect();

        //endianness??
        for idx in 0..256 {
            vocab.insert(idx, vec![idx]);  
        }

        for i in 0..num_merges{
            //let stats = get_stats(ids, None);
            let n;
            let pair: (u32, u32);
            {
                let mf = frequent_pair(ids.as_slice(), 1, Ordering::Descending);
                match mf.first(){
                    Some(val) => {
                        n = val.0;
                        pair = (val.1.0.clone(), val.1.1.clone())
                    },
                    None => {
                        println!("Break'd out of for i:{:?} in num_merges. No freq pair in ids: {:?}",i,ids);
                        break;
                    }
                }
            }
            let idx = 256 + i;
            ids = merge(&ids, &pair, &idx);

            merges.insert(pair.to_owned(), idx);
            //println!("inserted to merges: {:?}, {:?}\n\tmerges:{:?}\n",pair,idx,merges);
            vocab.insert(idx, [vocab.get(&pair.0).unwrap().clone(), vocab.get(&pair.1).unwrap().clone()].concat());

            if verbos {
                println!("merge {}/{}: {:?} -> {} ({:?} had {} occurrences)", i+1, num_merges, pair, idx, vocab[&idx], n)
            }

        }
        return BasicTokenizer{
            trained: true,
            vocab_size, num_merges, merges, vocab
        };
        
    }

    fn decode(&self, ids: Vec<u32>) -> String {
        let mut text_bytes= Vec::new();
        for id in ids {
            let word = self.vocab.get(&id).unwrap().clone();
            text_bytes = [text_bytes, word].concat();
        }
        let text8:String = text_bytes.iter().map(|x|{
            return char::from_u32(x.to_owned()).unwrap().to_string();
        }).collect();
        //let text = String::from_utf8_lossy(text8).into_owned();
        return text8
    }

    fn encode(&self, text:&String) -> Vec<u32> {
        let mut ids:Vec<u32> = text.chars().map(u32::from).collect();
        while ids.len() >= 2 {
            let pair;
            {
            let stats = frequent_pair(ids.as_slice(), 1, Ordering::Ascending);
            pair = (stats.first().unwrap().1.0.clone(), stats.first().unwrap().1.1.clone());
            }
            if !self.merges.contains_key(&pair){
                break;
            }
            let idx = self.merges.get(&pair).unwrap();
            ids = merge(&ids, &pair, &idx);

        }
        return ids
    }

    // format: split by lines 
    // vocab size
    // num_merges
    // merges seperated by ' ' then ',' first two -> 3rd
    // vocab seperated by ' ' then ',' first -> rest
    fn save(&self, path:&Path) -> Result<(), io::Error> {
        let mut model = String::new();
        model.push_str(format!("{}\n", self.vocab_size).as_str());
        model.push_str(format!("{}\n", self.num_merges).as_str());
        for merge in self.merges.borrow() {            
            model.push_str(format!("{},{},{} ", merge.0.0, merge.0.1, merge.1).as_str());
        }
        model.push('\n');
        for voc in self.vocab.borrow() {
            model.push_str(format!("{}", voc.0).as_str());
            for x in voc.1 {
                model.push_str(format!(",{}", x).as_str());
            }
            model.push(' ');
        }
//        model.push_str(format!("{:?}\n", self.merges).as_str());
//        model.push_str(format!("{:?}\n", self.vocab).as_str());

        println!("writing model as:\n{}", model);

        return fs::write(path, model);
    }

    fn load(path:&Path) -> Result<Self, String> {
        if let Ok(text) = fs::read_to_string(&path) {
            let lines:Vec<&str> = text.split('\n').collect();
            let vocab_size = lines.get(0).unwrap().parse::<u32>().unwrap();
            let num_merges = lines.get(1).unwrap().parse::<u32>().unwrap();
            let mut new_merges: HashMap<(u32,u32), u32> = HashMap::new(); 
            for merge in lines.get(2).unwrap().split(' '){
                let mut elems = merge.split(',');
                if let Ok(a) = elems.next().unwrap().parse::<u32>(){
                    if let Ok(b) = elems.next().unwrap().parse::<u32>(){
                        if let Ok(c) = elems.next().unwrap().parse::<u32>(){
                            //println!("parsed merge ({},{}) -> {}", a,b,c);
                            new_merges.insert((a,b), c);
                        } else { break; }                            
                    } else { break; }
                } else { break; }
            }
            let mut new_vocab: HashMap<u32, Vec<u32>> = HashMap::new();
            for voc in lines.get(3).unwrap().split(' '){
                let mut elems = voc.split(',');
                if let Ok(a) = elems.next().unwrap().parse::<u32>(){
                    let rv: Vec<Result<u32, String>> = elems
                    .map(|el|
                        match el.parse::<u32>(){
                            Ok(o) => {return Ok(o)},
                            Err(e) => {return Err(format!("Error parsing vocab ids: {}", el))}
                        })
                    .collect();
                    let v = rv.iter().map(|f|f.to_owned().unwrap()).collect();
                    //println!("parsed vocab {}, {:#?}",a,v);
                    new_vocab.insert(a, v);
                } else { break; }
            }
            return Ok(BasicTokenizer{vocab_size, trained:true, num_merges, merges:new_merges, vocab:new_vocab});
        }else {
            return Err(format!("Failed reading model from path: {}", path.to_str().unwrap()));
        }

    }
    fn load_mut(&mut self, path:&Path) -> Result<(), String> { 
        todo!()
    }
}

impl fmt::Display for BasicTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut out = Vec::new();
        write!(out,"BasicTokenizer:\n\t   trained: {}\n\tvocab_size: {}\n\tnum_merges: {}\n\tmerges:\n",self.trained,self.vocab_size,self.num_merges).unwrap();

        for elem in self.merges.borrow(){
            write!(out, "\t\t({:4},{:4}) -> {}\n", elem.0.0, elem.0.1, elem.1).unwrap();
        }

        write!(out,"\tvocab:\n", ).unwrap();

        for voc in self.vocab.borrow(){
            write!(out, "\t\t{:<4} : {:?}\n", voc.0, voc.1).unwrap()
        }

        return write!(f, "{}", String::from_utf8(out).unwrap());
    }
}


// todo 
//    cli argparse:
//          usg: ./app -e ./path.txt -m ./path.model -o ./path.ids(default stdout)
//          usg: ./app -d ./path.ids -m ./path.model -o ./path.txt(default stdout)
//          usg: ./app -t ./path.txt -o ./path.txt(default stdout)
// todo
//    repl: [e|enc|encode] ./path.txt (in) ([-o|o|out|output] ./out.ids)
//          [d|dec|decode] ./path.ids (in) ([-o|o|out|output] ./out.txt)
//          [t|tr|train] ./path.txt (in)
//          [p|pr|print] 
//          missing fn load [l|ld|load] ./path.model (in)
//          missing fn save [s|sv|save] ./path.model (out)

fn main() {
    let args:Vec<String> = args().collect();
    println!("Got args: {:?}", args);

    println!("repl usage: \n\t{}", usage());

    let model:Rc<RefCell<Option<BasicTokenizer>>> = Rc::new(RefCell::new(Option::None));
    let stdin = stdin();

    //TODO: use
    let not_exit = true;
    while not_exit{
        get_cmd(&stdin, model.clone())
    }

}

//ugly
fn usage() -> String {
    return "commands:
    \t[e|enc|encode] ./path.txt (in)
    \t[d|dec|decode] ./path.ids (in)
    \t[t|tr|train] ./path.txt (in)
    \t[l|ld|load] ./path.model (in)
    \t[s|sv|save] ./path.model (out)
    \t[p|pr|print]".to_string();
}

fn get_cmd(stdin:&Stdin, mut model:Rc<RefCell<Option<BasicTokenizer>>>){
    let mut line = String::new();
    print!("repl> ");
    stdout().flush();
    stdin.lock().read_line(&mut line).unwrap();

    match parse_line(&line) {
        Ok(REPLCommand::Decode(path)) => {
            match (*model).borrow().as_ref() {
                Some(tokenizer) => {
                    if let Ok(text) = fs::read_to_string(&path) {
                        let ids: Result<Vec<u32>, String> = text.split(',').map(|number|match number.parse::<u32>(){
                            Ok(o) => {return Ok(o)},
                            Err(err) => {Err(format!("Couldn't parse the file for ids. Expected format is comma seperated numbers: 1,2,3,4,... instead got: {}", number))},
                        }).collect();
                        match ids {
                            Ok(o) => {
                                //let ids = text.split(',').map(|number|match number.parse::<u32>()?{}).collect();
                                let result = tokenizer.decode(o);
                                println!("result:\n\t{}",result);
                            }
                            Err(e) => {
                                println!("{}",e);
                                return;
                            },
                        }
                    } else {
                        print!("Couldn't read file at {:?}", path);  
                    }
                }
                None => {
                    println!("Model is not initialized, train or load first")
                }
            }
        },
        Ok(REPLCommand::Encode(path)) => {
            match (*model).borrow().as_ref() {
                Some(tokenizer) => {
                    if let Ok(text) = fs::read_to_string(&path) {
                        let result = tokenizer.encode(&text);
                        println!("result:\n\t{:?}",result);
                    } else {
                        print!("Couldn't read file at {:?}", path);  
                    }                }
                None => {
                    println!("Model is not initialized, train or load first")
                }
            }
        },
        Ok(REPLCommand::Train(path)) => {
            if let Ok(text) = fs::read_to_string(&path) {
                let result = BasicTokenizer::train(&text, 512, Some(true));
                println!("result:\nmerges: {:?}\nvocab: {:?}", result.merges, result.vocab);
                *(*model).borrow_mut() = Some(result);
            } else {
                println!("Couldn't read file at {:?}", path);  
            }
        },
        Ok(REPLCommand::Print()) => {
            match (*model).borrow().as_ref() {
                Some(tokenizer) => {
                    println!("model:\n{}",tokenizer);
                },
                None => {
                    println!("Model is not initialized, train or load first")
                }
            }
        },
        Ok(REPLCommand::Save(path)) => {
            match (*model).borrow().as_ref() {
                Some(tokenizer) => {
                    println!("Writing model to path: {}\n", path.to_str().unwrap());
                    match tokenizer.save(&path) {
                        Ok(_) => {},
                        Err(e) => {println!("Failed writing with: {}", e)}   
                    }
                },
                None => {
                    println!("Model is not initialized, train or load first")
                }
            }
        },
        Ok(REPLCommand::Load(path)) => {
            println!("Loading model from path: {}\n", path.to_str().unwrap());
            match BasicTokenizer::load(&path) {
                Ok(new_tok) => {
                    *(*model).borrow_mut() = Some(new_tok);
                },
                Err(e) => {println!("Failed loading with: {}", e)}   
            }
        },
        Err(err) => {
            println!("{}",err)
        }
    }
}
        


fn parse_line(line:&str) -> Result<REPLCommand, String>{
    let args:Vec<&str> = line.split_whitespace().collect();

    // if !path.is_file(){
    //     return Err(format!("Parsed path isn't a file: {}\n\t{}", path.to_str().unwrap_or(""), path.));
    // }
        
    match args[0] {
        "e"|"enc"|"encode" => {
            if let Some(path) = args.get(1){
                return Ok(REPLCommand::Encode(Path::new(path.trim()).to_owned()))
            } else {
                return Err(format!("Not enough arguments for command {:?}\n{}", args, usage()))
            }
        },
        "d"|"dec"|"decode" => {
            if let Some(path) = args.get(1){
                return Ok(REPLCommand::Decode(Path::new(path.trim()).to_owned()))
            } else {
                return Err(format!("Not enough arguments for command {:?}\n{}", args, usage()))
            }
            },
        "t"|"tr"|"train" => {
            if let Some(path) = args.get(1){
                return Ok(REPLCommand::Train(Path::new(path.trim()).to_owned()))
            } else {
                return Err(format!("Not enough arguments for command {:?}\n{}", args, usage()))
            }
            }
        "p"|"pr"|"print" => {
            return Ok(REPLCommand::Print())
        },
        "s"|"sv"|"save" => {
            if let Some(path) = args.get(1){
                return Ok(REPLCommand::Save(Path::new(path.trim()).to_owned()))
            } else {
                return Err(format!("Not enough arguments for command {:?}\n{}", args, usage()))
            }                
        },
        "l"|"ld"|"load" => {
            if let Some(path) = args.get(1){
                return Ok(REPLCommand::Load(Path::new(path.trim()).to_owned()))
            } else {
                return Err(format!("Not enough arguments for command {:?}\n{}", args, usage()))
            }                
        },
        _ => {
            return Err(format!("Couldn't parse \"{}\" into a command, expected: {}", args[0], usage()))
        }
    }
}

enum REPLCommand  {
    Encode(PathBuf),
    Decode(PathBuf),
    Train(PathBuf),
    Print(),
    Save(PathBuf),
    Load(PathBuf)
}