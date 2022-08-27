use std::collections::BinaryHeap;

use crate::{
    neurons::{self, NetworkDescription},
    parallel::ForkJoinPool,
};

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

pub trait Input: Send {
    fn get_network_inputs(&self) -> &Vec<f64>;
}

pub trait Scorer<InputType: 'static + Input + Clone>: Send + Clone {
    fn score(&self, input: &InputType, outputs: &Vec<f64>) -> f64;

    fn create_worker_pool() -> ForkJoinPool<InputType, f64, (Self, neurons::Network, fn(f64) -> f64)>
    where
        Self: 'static,
    {
        let pool = ForkJoinPool::new(
            |(scorer, mut network, activation_function): (
                Self,
                neurons::Network,
                fn(f64) -> f64,
            ),
             input: InputType| {
                scorer.score(
                    &input,
                    &network.apply(input.get_network_inputs(), activation_function),
                )
            },
        );
        pool
    }

    fn score_network(
        &self,
        inputs: &Vec<InputType>,
        network: &mut neurons::Network,
        activation_function: neurons::ActivationFunction,
        thread_pool: &mut ForkJoinPool<InputType, f64, (Self, neurons::Network, fn(f64) -> f64)>,
    ) -> f64
    where
        Self: 'static + Sized,
    {
        let total_score = ForkJoinPool::exec_and_collect(
            thread_pool,
            inputs.clone(),
            &|| 0.0,
            &|a, b| a + b,
            (self.clone(), network.clone(), activation_function),
        );
        total_score / inputs.len() as f64
    }
}

impl ScoredNetwork {
    pub fn new<InputType: 'static + Input + Clone, ScorerType: 'static + Scorer<InputType>>(
        mut network: neurons::Network,
        inputs: &Vec<InputType>,
        scorer: &ScorerType,
        activation_function: neurons::ActivationFunction,
        thread_pool: &mut ForkJoinPool<
            InputType,
            f64,
            (ScorerType, neurons::Network, fn(f64) -> f64),
        >,
    ) -> ScoredNetwork {
        let total_score =
            scorer.score_network(inputs, &mut network, activation_function, thread_pool);
        ScoredNetwork {
            score: total_score / inputs.len() as f64,
            network: network,
            age: 1.0,
        }
    }
}

pub struct Trainer<InputType: 'static + Input + Clone, ScorerType: Scorer<InputType>> {
    networks: BinaryHeap<ScoredNetwork>,
    population_size: usize,
    pub inputs: Vec<InputType>,
    pub scorer: ScorerType,
}

impl<InputType: Input + Clone, ScorerType: 'static + Scorer<InputType>>
    Trainer<InputType, ScorerType>
{
    pub fn new(
        population_size: usize,
        network_description: &NetworkDescription,
        inputs: Vec<InputType>,
        scorer: ScorerType,
        activation_function: neurons::ActivationFunction,
        thread_pool: &mut ForkJoinPool<
            InputType,
            f64,
            (ScorerType, neurons::Network, fn(f64) -> f64),
        >,
    ) -> Trainer<InputType, ScorerType> {
        let mut networks: BinaryHeap<ScoredNetwork> = BinaryHeap::with_capacity(population_size);
        for _i in 0..population_size {
            networks.push(ScoredNetwork::new(
                neurons::randomized_network(network_description),
                &inputs,
                &scorer,
                activation_function,
                thread_pool,
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
        activation_function: neurons::ActivationFunction,
        thread_pool: &mut ForkJoinPool<
            InputType,
            f64,
            (ScorerType, neurons::Network, fn(f64) -> f64)>,
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
                    thread_pool,
                ));
                network.age += 0.125;
                new_networks.push(network);
            }
            self.networks = new_networks;
        }
    }
}
