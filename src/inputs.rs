use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::trainer;

#[derive(Debug, Clone)]
pub struct Input {
    pub value: String,
    pub network_inputs: Vec<f64>,
    pub correctly_spelled: bool,
}

impl trainer::Input for Input {
    fn get_network_inputs(&self) -> &Vec<f64> {
        &self.network_inputs
    }
}

#[derive(Clone, Copy)]
pub struct Scorer {}
impl trainer::Scorer<Input> for Scorer {
    fn score(&self, input: &Input, outputs: &Vec<f64>) -> f64 {
        if input.correctly_spelled {
            outputs[0].round()
        } else {
            1.0 - outputs[0].round()
        }
    }
}

pub fn convert_to_network_inputs(value: &str) -> Vec<f64> {
    let mut result = Vec::with_capacity(value.len());
    for i in 0..20 {
        if i < value.len() {
            result.push(convert_to_network_input(value.chars().nth(i).unwrap()));
        } else {
            result.push(0.0);
        }
    }
    result
}
pub fn convert_to_network_input(c: char) -> f64 {
    if c == ' ' {
        return 0.0;
    } else if c.is_lowercase() {
        // The number in the alphabet. For example, a = 1, b = 2, z = 26.
        let number = c as u8 - 96u8;
        return number as f64;
    } else {
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
        let network_inputs = convert_to_network_inputs(value.as_str());
        inputs.push(Input {
            value: value,
            network_inputs: network_inputs,
            correctly_spelled: correctly_spelled,
        });
    }
    inputs
}
