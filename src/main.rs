#![allow(dead_code)]

extern crate rand;
mod inputs;
mod neurons;

fn main() {
    let mut network = neurons::randomized_network(&neurons::NetworkDescription {
        layer_sizes: vec![20, 20, 1],
    });
    let inputs = inputs::read_inputs("inputs.txt");
    for input in inputs {
        let results = network.apply(&input.network_inputs, &default_activation);
        println!("{}: {}", input.value, results[0]);
    }
}

fn default_activation(x: f64) -> f64 {
        (x / (x.abs() + 1.0) + 1.0) / 2.0
}
