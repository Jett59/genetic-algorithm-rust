use std::{
    fs::File,
    io::{BufRead, BufReader},
};

#[derive(Debug)]
pub struct Input {
    pub value: String,
    pub network_inputs: Vec<f64>,
    pub correctly_spelled: bool,
}

pub fn convert_to_network_input(c: char)-> f64 {
    if c == ' ' {
        return 0.0;
    }else if c.is_lowercase() {
    // The number in the alphabet. For example, a = 1, b = 2, z = 26.
    let number = c as u8 - 96u8;
    return number as f64;
    }else {
        return 0.0;
    }
}

pub fn read_inputs(path: &str) -> Vec<Input> {
    let mut inputs: Vec<Input> = Vec::new();
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let mut words = line.split_whitespace();
        let value = words.next().unwrap().to_string();
        let correctly_spelled = words.next().unwrap().to_string() == "true";
        let mut network_inputs: Vec<f64> = Vec::new();
        for i in 0..20 {
            if i < value.len() {
                network_inputs.push(convert_to_network_input(value.chars().nth(i).unwrap()));
            } else {
                network_inputs.push(0.0);
            }
        }
        inputs.push(Input {
            value: value,
            network_inputs: network_inputs,
            correctly_spelled: correctly_spelled,
        });
    }
    inputs
}
