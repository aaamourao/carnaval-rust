use ndarray::{Array, Ix3};

pub mod conv;

pub trait NeuralNetwork {
    fn forward(input: Array<f64, Ix3>,) -> Array<f64, Ix3>;
}
