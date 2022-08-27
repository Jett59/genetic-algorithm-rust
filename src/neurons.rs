use rand::{thread_rng, Rng};

#[derive(Clone)]
pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub value: f64, // Temperary storage for the value of the neuron.
}
unsafe impl Send for Neuron {}

#[derive(Clone)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}
unsafe impl Send for Layer {}

#[derive(Clone)]
pub struct Network {
    pub layers: Vec<Layer>,
}
unsafe impl Send for Network {}

pub struct NetworkDescription {
    pub layer_sizes: Vec<usize>,
}
unsafe impl Send for NetworkDescription {}

pub fn randomized_network(description: &NetworkDescription) -> Network {
    let mut network = Network { layers: Vec::new() };
    for layer_size_index in 0..description.layer_sizes.len() - 1 {
        let layer_size = description.layer_sizes[layer_size_index];
        let next_layer_size = description.layer_sizes[layer_size_index + 1];
        let mut layer = Layer {
            neurons: Vec::new(),
        };
        for _i in 0..layer_size {
            let mut neuron = Neuron {
                weights: Vec::new(),
                bias: thread_rng().gen(),
                value: 0.0,
            };
            for _j in 0..next_layer_size {
                neuron.weights.push(thread_rng().gen());
            }
            layer.neurons.push(neuron);
        }
        network.layers.push(layer);
    }
    let last_layer_size = description.layer_sizes[description.layer_sizes.len() - 1];
    let mut last_layer = Layer {
        neurons: Vec::new(),
    };
    for _i in 0..last_layer_size {
        let neuron = Neuron {
            weights: Vec::new(),
            bias: thread_rng().gen(),
            value: 0.0,
        };
        last_layer.neurons.push(neuron);
    }
    network.layers.push(last_layer);
    network
}

pub type ActivationFunction = fn(f64) -> f64;

pub fn default_activation(x: f64) -> f64 {
    (x / (x.abs() + 1.0) + 1.0) / 2.0
}

impl Network {
    pub fn apply(
        &mut self,
        inputs: &Vec<f64>,
        activation_function: ActivationFunction,
    ) -> Vec<f64> {
        {
            let first_layer = &mut self.layers[0];
            assert!(first_layer.neurons.len() == inputs.len());
            for i in 0..first_layer.neurons.len() {
                first_layer.neurons[i].value = inputs[i];
            }
        }
        for layer_index in 0..self.layers.len() - 1 {
            // Silly rust doesn't allow us to access two elements of the vector without doing something like this.
            let layers_split = self.layers.split_at_mut(layer_index + 1);
            let layer = &mut layers_split.0[layer_index];
            let next_layer = &mut layers_split.1[0];
            for neuron in &mut layer.neurons {
                neuron.value = activation_function(neuron.bias + neuron.value);
                for weight_index in 0..neuron.weights.len() {
                    next_layer.neurons[weight_index].value +=
                        neuron.weights[weight_index] * neuron.value;
                }
            }
        }
        {
            let last_layer = &self.layers[self.layers.len() - 1];
            let mut outputs: Vec<f64> = Vec::with_capacity(last_layer.neurons.len());
            for neuron in &last_layer.neurons {
                outputs.push(activation_function(neuron.value));
            }
            self.clear();
            return outputs;
        }
    }
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            for neuron in &mut layer.neurons {
                neuron.value = 0.0;
            }
        }
    }
    pub fn mutated(&self, mutation_rate: f64) -> Network {
        let mut network = self.clone();
        if network.layers.len() > 0 {
            let layer_index = thread_rng().gen_range(0..network.layers.len());
            let layer = &mut network.layers[layer_index];
            if layer.neurons.len() > 0 {
                let neuron_index = thread_rng().gen_range(0..layer.neurons.len());
                let neuron = &mut layer.neurons[neuron_index];
                if neuron.weights.len() > 0 {
                    if thread_rng().gen_bool(0.9) {
                        let weight_index = thread_rng().gen_range(0..neuron.weights.len());
                        let weight = &mut neuron.weights[weight_index];
                        *weight += thread_rng().gen_range(-1.0..1.0) * mutation_rate;
                    }
                }
                if thread_rng().gen_bool(0.25) {
                    neuron.bias += thread_rng().gen_range(-1.0..1.0) * mutation_rate;
                }
            }
        }
        network
    }
}
