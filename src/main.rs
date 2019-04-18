extern crate rand;

use rand::Rng;

static LAMBDA: f64 = 1.0;
static MAX_LEARNING_EPOCH: usize = 100000;

static MOMENTUM: f64 = 1.0;

#[derive(Debug)]
struct Network {
    pub layers: Vec<Vec<Neuron>>,
    alpha: f64,
}

impl Network {
    pub fn new(alpha: f64, layout: Vec<usize>, inputs: usize) -> Network {
        // create network
        let mut network = Network {
            alpha: alpha,
            layers: Vec::with_capacity(layout.len()),
        };
        // fill with neurons
        // input layer
        network.layers.push(Vec::with_capacity(layout[0]));
        for _ in 0..layout[0] {
            network.layers[0].push(Neuron::new(inputs));
        }
        // remaining layers
        if layout.len() > 1 {
            for i in 1..layout.len() {
                network.layers.push(Vec::with_capacity(layout[i]));
                for _ in 0..layout[i] {
                    network.layers[i].push(Neuron::new(layout[i - 1])); // each neuron has input count of previous layer neuron count
                }
            }
        }
        return network;
    }

    fn calculate_layer_result(&mut self, x: &Vec<f64>, layer: usize) -> Vec<f64> {
        let mut result = Vec::<f64>::with_capacity(self.layers[layer].len());
        for neuron in self.layers[layer].iter_mut() {
            result.push(neuron.calculate_output(&x));
        }
        return result;
    }

    pub fn calculate_network_output(&mut self, x: &Vec<f64>) -> Vec<f64> {
        let mut last_layer_result = self.calculate_layer_result(x, 0);
        if self.layers.len() > 1 {
            for i in 1..self.layers.len() {
                last_layer_result = self.calculate_layer_result(&last_layer_result, i);
            }
        }
        return last_layer_result;
    }

    // calculates network output, but remembers all outputs and inputs of all neurons
    pub fn calculate_network_output_full(&mut self, x: &Vec<f64>) -> Vec<Vec<f64>> {
        let mut ret = Vec::with_capacity(self.layers.len());
        ret.push(self.calculate_layer_result(x, 0));

        for i in 1..self.layers.len() {
            ret.push(Vec::with_capacity(self.layers[i].len()));
            ret[i] = self.calculate_layer_result(&ret[i - 1], i);
        }
        return ret;
    }

    pub fn recalculate_weights(
        &mut self,
        input: &Vec<f64>,
        received: &Vec<f64>,
        expected: &Vec<f64>,
        all_neuron_outs: &Vec<Vec<f64>>,
    ) {
        let mut errors: Vec<Vec<f64>> = Vec::new();

        // output layer
        errors.push(Vec::new());
        for (i, _neuron) in self.layers.last().unwrap().iter().enumerate() {
            // neuron error
            let e = (expected[i] - received[i]) * Neuron::derived_activation(received[i]);
            errors[0].push(e);
        }

        // other layers
        if self.layers.len() > 1 {
            for layer_number in (0..self.layers.len() - 1).rev() {
                // from last layer to input
                errors.push(Vec::new());
                for (neuron_num, _neuron) in self.layers[layer_number].iter().enumerate() {
                    let mut e = 0.0;
                    let errors_last = errors.len() - 1;
                    for i in 0..self.layers[layer_number + 1].len() {
                        // for every neuron in next layer
                        e += self.layers[layer_number + 1][i].weights[neuron_num]
                            * errors[errors_last - 1][i];
                    }
                    e *= Neuron::derived_activation(all_neuron_outs[layer_number][neuron_num]);
                    errors[errors_last].push(e);
                }
            }
        }

        errors.reverse(); // we put backwards, so now reverse

        // recalculate weights for first layer
        for (neuron_num, neuron) in self.layers[0].iter_mut().enumerate() {
            for (weight_num, weight) in neuron.weights.iter_mut().enumerate() {
                *weight =
                    MOMENTUM * (*weight + self.alpha * errors[0][neuron_num] * input[weight_num]);
            }
            neuron.bias = MOMENTUM * (neuron.bias + self.alpha * errors[0][neuron_num]);
        }

        // recalculate weights for all other layers
        for layer_num in 1..self.layers.len() {
            for (neuron_num, neuron) in self.layers[layer_num].iter_mut().enumerate() {
                for (weight_num, weight) in neuron.weights.iter_mut().enumerate() {
                    *weight = MOMENTUM
                        * (*weight
                            + self.alpha
                                * errors[layer_num][neuron_num]
                                * all_neuron_outs[layer_num - 1][weight_num]);
                }
                neuron.bias = MOMENTUM * (neuron.bias + self.alpha * errors[layer_num][neuron_num]);
            }
        }
    }

    pub fn calculate_network_error(expected: &Vec<f64>, received: &Vec<f64>) -> f64 {
        let mut net_error = 0.0;
        for i in 0..expected.len() {
            net_error += (expected[i] - received[i]).powf(2.0);
        }
        return net_error / 2.0;
    }

    pub fn start_learning(&mut self, mut training_set: Vec<TrainingData>, stop_error: f64) {
        let mut rng = rand::thread_rng();
        for i in 0..MAX_LEARNING_EPOCH {
            println!("Epoch {}", i);
            let mut epoch_error = 0.0;
            for training in training_set.iter() {
                let out = self.calculate_network_output_full(&training.inputs);
                self.recalculate_weights(
                    &training.inputs,
                    out.last().unwrap(),
                    &training.expected,
                    &out,
                );
                epoch_error +=
                    Network::calculate_network_error(&training.expected, &out.last().unwrap());
            }
            rng.shuffle(&mut training_set);
            println!("Epoch error: {}", epoch_error);
            if epoch_error < stop_error {
                break;
            }
        }
        println!("Learning completed");
    }
}

#[test]
fn test_network_new() {
    let network = Network::new(1.0, vec![3], 24);
    // network layers count
    assert_eq!(network.layers.len(), 1);
    // neurons in layer
    assert_eq!(network.layers[0].len(), 3);
    // amount of inputs
    assert_eq!(network.layers[0][0].weights.len(), 24);

    let network = Network::new(1.0, vec![3, 5], 2);
    // network layers count
    assert_eq!(network.layers.len(), 2);
    // neurons in layer
    assert_eq!(network.layers[0].len(), 3);
    assert_eq!(network.layers[1].len(), 5);
    // amount of inputs
    assert_eq!(network.layers[0][2].weights.len(), 2);
    assert_eq!(network.layers[1][3].weights.len(), 3);

    let network = Network::new(1.0, vec![3, 5, 2], 3);
    // network layers count
    assert_eq!(network.layers.len(), 3);
    // neurons in layer
    assert_eq!(network.layers[0].len(), 3);
    assert_eq!(network.layers[2].len(), 2);
    // amount of inputs
    assert_eq!(network.layers[0][2].weights.len(), 3);
    assert_eq!(network.layers[1][3].weights.len(), 3);
    assert_eq!(network.layers[2][1].weights.len(), 5);
}

#[test]
fn test_calculate_layer_result() {
    let mut network = Network::new(1.0, vec![1], 2);
    network.layers[0][0].weights = vec![1.0, 2.0];
    network.layers[0][0].bias = -1.0;

    assert_eq!(network.calculate_layer_result(&vec![3.0, -1.0], 0), [0.5]);
}

#[test]
fn test_calculate_network_output() {
    let mut network = Network::new(1.0, vec![2, 1], 2);
    network.layers[0][0].weights = vec![1.0, 1.0];
    network.layers[0][0].bias = 0.0;
    network.layers[0][1].weights = vec![1.0, 1.0];
    network.layers[0][1].bias = 0.0;
    network.layers[1][0].weights = vec![1.0, 1.0];
    network.layers[1][0].bias = 0.0;

    assert_eq!(
        network.calculate_network_output(&vec![0.0, 0.0]),
        [0.7310585786300049]
    );
}

#[test]
fn test_calculate_network_output_full() {
    let mut network = Network::new(1.0, vec![2, 1], 2);
    network.layers[0][0].weights = vec![1.0, 1.0];
    network.layers[0][0].bias = 0.0;
    network.layers[0][1].weights = vec![1.0, 1.0];
    network.layers[0][1].bias = 0.0;
    network.layers[1][0].weights = vec![1.0, 1.0];
    network.layers[1][0].bias = 0.0;

    assert_eq!(
        network.calculate_network_output_full(&vec![0.0, 0.0]),
        vec![vec![0.5, 0.5], vec![0.7310585786300049]]
    );
}

#[test]
fn test_recalculate_weights() {
    let mut network = Network::new(1.0, vec![2, 1], 2);
    network.layers[0][0].weights = vec![1.0, 1.0];
    network.layers[0][0].bias = 0.0;
    network.layers[0][1].weights = vec![1.0, 1.0];
    network.layers[0][1].bias = 0.0;
    network.layers[1][0].weights = vec![1.0, 1.0];
    network.layers[1][0].bias = 0.0;

    let input = vec![0.0, 0.0];
    let full_output = network.calculate_network_output_full(&input);
    network.recalculate_weights(
        &input,
        full_output.last().unwrap(),
        &vec![0.0],
        &full_output,
    );

    assert_eq!(network.layers[0][0].weights, vec![1.0, 1.0]);
    assert_eq!(network.layers[0][1].weights, vec![1.0, 1.0]);
    assert_eq!(
        network.layers[1][0].weights,
        vec![0.9198168137456808, 0.9198168137456808]
    );
    assert_eq!(network.layers[0][0].bias, -0.037686692851833764);
    assert_eq!(network.layers[0][1].bias, -0.037686692851833764);
    assert_eq!(network.layers[1][0].bias, -0.16036637250863844);
}

#[test]
fn test_calculate_network_error() {
    let err = Network::calculate_network_error(&vec![0.0, 1.0], &vec![0.0, 0.0]);
    assert_eq!(err, 0.5);
}

#[derive(Debug)]
struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Neuron {
    pub fn new(inputs: usize) -> Neuron {
        let mut rng = rand::thread_rng();

        let mut neuron = Neuron {
            weights: Vec::with_capacity(inputs),
            bias: rng.gen_range(-10.0, 10.0),
        };

        for _ in 0..inputs {
            neuron.weights.push(rng.gen_range(-10.0, 10.0));
        }

        return neuron;
    }

    /// unipolar sigmoid
    pub fn activation(x: f64) -> f64 {
        return 1.0 / (1.0 + std::f64::consts::E.powf(-LAMBDA * x));
    }

    /// derivation of unipolar sigmoid
    pub fn derived_activation(x: f64) -> f64 {
        let act_x = Neuron::activation(x);
        return LAMBDA * act_x * (1.0 - act_x);
    }

    // calculate neuron output
    pub fn calculate_output(&mut self, x: &Vec<f64>) -> f64 {
        assert!(x.len() == self.weights.len());
        let mut out = 0.0;
        for i in 0..x.len() {
            out += x[i] * self.weights[i];
        }
        out += self.bias;
        let result = Neuron::activation(out);
        return result;
    }
}

#[test]
fn test_neuron_new() {
    // make sure neurons are created with correct input count
    let neuron = Neuron::new(10);
    assert_eq!(neuron.weights.len(), 10);
}

#[test]
fn test_neuron_activation() {
    // check unipolar sigmoid result values
    assert_eq!(Neuron::activation(0.0), 0.5);
    assert!(Neuron::activation(10.0) > 0.95);
    assert!(Neuron::activation(-10.0) < 0.05);
}

#[test]
fn test_neuron_derived_activation() {
    // check derivation values
    for i in -10..10 {
        assert!(Neuron::derived_activation(i as f64) > 0.0);
    }
}

#[test]
fn test_neuron_calculate_output() {
    let mut neuron = Neuron::new(2);
    neuron.weights = vec![1.0, 2.0];
    neuron.bias = -1.0;
    assert_eq!(neuron.calculate_output(&vec![2.0, 1.0]), 0.9525741268224331);
    assert_eq!(neuron.calculate_output(&vec![3.0, -1.0]), 0.5);
}

#[derive(Debug)]
struct TrainingData {
    inputs: Vec<f64>,
    expected: Vec<f64>,
}

impl TrainingData {
    pub fn generate_for_xor() -> Vec<TrainingData> {
        let mut ret = Vec::<TrainingData>::with_capacity(4);
        ret.push(TrainingData {
            inputs: vec![0.0, 0.0],
            expected: vec![0.0],
        });
        ret.push(TrainingData {
            inputs: vec![1.0, 0.0],
            expected: vec![1.0],
        });
        ret.push(TrainingData {
            inputs: vec![0.0, 1.0],
            expected: vec![1.0],
        });
        ret.push(TrainingData {
            inputs: vec![1.0, 1.0],
            expected: vec![0.0],
        });
        return ret;
    }
}

fn main() {
    let mut network = Network::new(0.3, vec![2, 1], 2);
    let training_data = TrainingData::generate_for_xor();
    network.start_learning(training_data, 0.01);
    let verification_data = TrainingData::generate_for_xor();
    println!(
        "Should be 0: {:?}",
        network.calculate_network_output(&verification_data[0].inputs)
    );
    println!(
        "Should be 1: {:?}",
        network.calculate_network_output(&verification_data[1].inputs)
    );
    println!(
        "Should be 1: {:?}",
        network.calculate_network_output(&verification_data[2].inputs)
    );
    println!(
        "Should be 0: {:?}",
        network.calculate_network_output(&verification_data[3].inputs)
    );
}
