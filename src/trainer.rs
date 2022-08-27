use std::collections::BinaryHeap;

use crate::neurons::{self, NetworkDescription};

#[derive(Clone)]
pub struct ScoredNetwork {
    pub score: f64,
    pub network: neurons::Network,
    age: f64,
}

impl PartialEq for ScoredNetwork {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for ScoredNetwork {}

impl PartialOrd for ScoredNetwork {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (self.score / self.age as f64).partial_cmp(&(other.score / other.age as f64))
    }
}
impl Ord for ScoredNetwork {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}

pub trait Input {
    fn get_network_inputs(&self) -> &Vec<f64>;
}

pub trait Scorer<InputType: Input> {
    fn score(&self, input: &InputType, outputs: &Vec<f64>) -> f64;

    fn score_network(
        &self,
        inputs: &Vec<InputType>,
        network: &mut neurons::Network,
        activation_function: &neurons::ActivationFunction,
    ) -> f64 {
        let mut score = 0.0;
        for input in inputs {
            let output = network.apply(input.get_network_inputs(), activation_function);
            score += self.score(input, &output);
        }
        score / inputs.len() as f64
    }
}

impl ScoredNetwork {
    pub fn new<InputType: Input, ScorerType: Scorer<InputType>>(
        mut network: neurons::Network,
        inputs: &Vec<InputType>,
        scorer: &ScorerType,
        activation_function: &neurons::ActivationFunction,
    ) -> ScoredNetwork {
        let mut total_score = 0.0;
        for input in inputs {
            let outputs = network.apply(input.get_network_inputs(), activation_function);
            total_score += scorer.score(input, &outputs);
        }
        ScoredNetwork {
            score: total_score / inputs.len() as f64,
            network: network,
            age: 1.0,
        }
    }
}

pub struct Trainer<InputType: Input, ScorerType: Scorer<InputType>> {
    networks: BinaryHeap<ScoredNetwork>,
    population_size: usize,
    pub inputs: Vec<InputType>,
    pub scorer: ScorerType,
}

impl<InputType: Input, ScorerType: Scorer<InputType>> Trainer<InputType, ScorerType> {
    pub fn new(
        population_size: usize,
        network_description: &NetworkDescription,
        inputs: Vec<InputType>,
        scorer: ScorerType,
        activation_function: &neurons::ActivationFunction,
    ) -> Trainer<InputType, ScorerType> {
        let mut networks: BinaryHeap<ScoredNetwork> = BinaryHeap::with_capacity(population_size);
        for _i in 0..population_size {
            networks.push(ScoredNetwork::new(
                neurons::randomized_network(network_description),
                &inputs,
                &scorer,
                activation_function,
            ));
        }
        Trainer {
            networks: networks,
            population_size,
            inputs: inputs,
            scorer: scorer,
        }
    }
    pub fn get_best(&mut self) -> ScoredNetwork {
        (*self
            .networks
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .unwrap())
        .clone()
    }
    pub fn train(
        &mut self,
        mutation_rate: f64,
        iterations: usize,
        activation_function: &neurons::ActivationFunction,
    ) {
        for _i in 0..iterations {
            let mut new_networks: BinaryHeap<ScoredNetwork> =
                BinaryHeap::with_capacity(self.population_size);
            while new_networks.len() < self.population_size {
                let mut network = self.networks.pop().unwrap();
                new_networks.push(ScoredNetwork::new(
                    network.network.mutated(mutation_rate),
                    &self.inputs,
                    &self.scorer,
                    activation_function,
                ));
                network.age += 0.125;
                new_networks.push(network);
            }
            self.networks = new_networks;
        }
    }
}
