// activation functions

use std::f64;
use ndarray::{Array, Ix3};

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum ActivationFunctionType {
    None,
    Sigmoid,
    Relu,
    LeakyRelu,
    Tanh,
    Softmax,
}

pub fn sigmoid(x: &f64) -> f64 {
    let minus_x = -x;
    1.0 / (1.0 + minus_x.exp())
}

pub fn relu(x: &f64) -> f64 {
    x.max(0.0)
}

pub fn leaky_relu(x: &f64, alpha: Option<f64>) -> f64 {
    let leak = alpha.unwrap_or_else(|| 0.1) * x;
    x.max(leak)
}

pub fn tanh(x: &f64) -> f64 {
    x.clone().tanh()
}

pub fn softmax(x: &Array<f64, Ix3>) -> Array<f64, Ix3> {
    let exp_scores = x.clone().map(|x| x.exp());
    exp_scores.clone() / exp_scores.sum()
}