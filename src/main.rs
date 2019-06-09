extern crate rand;
extern crate multilayer;

use multilayer::{Network, TrainingData};

fn main() {
    let mut network = Network::new(0.1, vec![2, 1], 2);
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
