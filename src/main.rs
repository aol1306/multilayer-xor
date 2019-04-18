extern crate rand;

use rand::Rng;

static LAMBDA: f64 = 1.0;

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
                    network.layers[i].push(Neuron::new(layout[i-1])); // each neuron has input count of previous layer neuron count
                }
            }
        }
        return network;
    }

    fn calculate_layer_result(&self, x: Vec<f64>, layer: &Vec<Neuron>) -> Vec<f64> {
        let mut result = Vec::<f64>::with_capacity(layer.len());
        for neuron in layer {
            result.push(neuron.calculate_output(&x));
        }
        return result;
    }

    fn calculate_network_output(&self, x: Vec<f64>) -> Vec<f64> {
        let mut last_layer_result = self.calculate_layer_result(x, &self.layers[0]);
        if self.layers.len() > 1 {
            for i in 1..self.layers.len() {
                last_layer_result = self.calculate_layer_result(last_layer_result, &self.layers[i]);
            }
        }
        return last_layer_result;
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

    assert_eq!(network.calculate_layer_result(vec![3.0, -1.0], &network.layers[0]), [0.5]);
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

    assert_eq!(network.calculate_network_output(vec![0.0, 0.0]), [0.7310585786300049]);
}

#[derive(Debug)]
struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub last_result: f64,
}

impl Neuron {
    pub fn new(inputs: usize) -> Neuron {
        let mut rng = rand::thread_rng();

        let mut neuron = Neuron {
            weights: Vec::with_capacity(inputs),
            bias: rng.gen_range(-10.0, 10.0),
            last_result: 0.0,
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

    // calculate network output
    pub fn calculate_output(&self, x: &Vec<f64>) -> f64 {
        assert!(x.len() == self.weights.len());
        let mut out = 0.0;
        for i in 0..x.len() {
            out += x[i] * self.weights[i];
        }
        out += self.bias;
        let result = Neuron::activation(out);
        //self.last_result = result;
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
    println!("{:?}", TrainingData::generate_for_xor());
    println!("{:?}", Neuron::new(5));
}
