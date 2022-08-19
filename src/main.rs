#![allow(dead_code)]

extern crate rand;
mod neurons;
mod inputs;

fn main() {
    println!("{:#?}", inputs::read_inputs("inputs.txt"));
}
