#![allow(dead_code)]

use std::io::Write;

use rand::{seq::SliceRandom, thread_rng};
use trainer::{Scorer, Trainer};

mod inputs;
mod neurons;
mod trainer;

fn main() {
    let network_description = neurons::NetworkDescription {
        layer_sizes: vec![20, 25, 25, 1],
    };
    let mut inputs = inputs::read_inputs("inputs.txt");
    inputs.shuffle(&mut thread_rng());
    let (training_inputs, testing_inputs) = inputs.split_at(inputs.len() / 2);
    let scorer = inputs::Scorer {};
    let mut trainer = Trainer::new(
        1024,
        &network_description,
        training_inputs.to_vec(),
        scorer,
        &neurons::default_activation,
    );
    println!("Training score: {}", trainer.get_best().score);
    println!(
        "Testing score: {}",
        scorer.score_network(
            &testing_inputs.to_vec(),
            &mut trainer.get_best().network,
            &neurons::default_activation
        )
    );
    trainer.train(1000.0, 128, &neurons::default_activation);
    println!("Training score: {}", trainer.get_best().score);
    println!(
        "Testing score: {}",
        scorer.score_network(
            &testing_inputs.to_vec(),
            &mut trainer.get_best().network,
            &neurons::default_activation
        )
    );
    loop {
        print!("> ");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        std::io::stdin()
            .read_line(&mut line)
            .expect("Failed to read from standard input");
        line = String::from(line.trim());
        if line == "exit" {
            break;
        } else if line == "train" {
            trainer.train(1000.0, 128, &neurons::default_activation);
            println!("Training score: {}", trainer.get_best().score);
            println!(
        "Testing score: {}",
        scorer.score_network(
            &testing_inputs.to_vec(),
            &mut trainer.get_best().network,
            &neurons::default_activation
        )
    );
        } else {
            let output = trainer.get_best().network.apply(
                &inputs::convert_to_network_inputs(line.as_str()),
                &neurons::default_activation,
            )[0];
            if output < 0.5 {
                println!("{} is not correctly spelled", line);
            } else {
                println!("{} is correctly spelled", line);
            }
        }
    }
}
