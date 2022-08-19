use rand::{thread_rng, Rng};

#[derive(Clone)]
pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub value: f64, // Temperary storage for the value of the neuron.
}
#[derive(Clone)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}
#[derive(Clone)]
pub struct Network {
    pub layers: Vec<Layer>,
}

pub struct NetworkDescription {
    pub layer_sizes: Vec<usize>,
}

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

type ActivationFunction = dyn Fn(f64) -> f64;

impl Network {
    pub fn apply(
        &mut self,
        inputs: &Vec<f64>,
        activation_function: &ActivationFunction,
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
}
