use std::borrow::{Borrow, BorrowMut};
use std::cell::RefCell;
use std::char::REPLACEMENT_CHARACTER;
use std::ffi::OsString;
use std::{fs, path};
use std::io::{self, stdin, stdout, BufRead, Stdin, Write};
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
    
    fn save(&self, path:&Path) -> Result<(), String> {
        todo!()
    }
    fn load(&self, path:&Path) -> Result<Self, String> {
        todo!()
    }
    fn load_mut(&mut self, path:&Path) -> Result<(), String> { 
        todo!()
    }
}


// todo 
//    cli argparse:
//          usg: ./app -e ./path.txt -m ./path.model -o ./path.ids(default stdout)
//          usg: ./app -d ./path.ids -m ./path.model -o ./path.txt(default stdout)
//          usg: ./app -t ./path.txt -o ./path.txt(default stdout)
// todo
//    repl: [e|enc|encode] ./path.txt (in)
//          [d|dec|decode] ./path.ids (in)
//          [t|tr|train] ./path.txt (in)
//          missing fn load [l|ld|load] ./path.model (in)
//          missing fn save [s|sv|save] ./path.model (out)

fn main() {
    println!("{}", usage());

    let model:Rc<RefCell<Option<BasicTokenizer>>> = Rc::new(RefCell::new(Option::None));
    let stdin = stdin();

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
                            Err(err) => {Err(format!("Couldn't parse the file for ids. Expected format is comma seperated numbers 1,2,3,4,... instead got: {}", number))},
                        }).collect();
                        match ids {
                            Ok(o) => {
                                //let ids = text.split(',').map(|number|match number.parse::<u32>()?{}).collect();
                                let result = tokenizer.decode(o);
                                println!("result:\n\t{}",result);
                            }
                            Err(e) => return,
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
                    println!("model:\n{:?}",tokenizer);
                },
                None => {
                    println!("Model is not initialized, train or load first")
                }
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
            _ => {
                return Err(format!("Couldn't parse \"{}\" into a command, expected: {}", args[0], usage()))
            }
    }

}

enum REPLCommand  {
    Encode(PathBuf),
    Decode(PathBuf),
    Train(PathBuf),
    Print()
}