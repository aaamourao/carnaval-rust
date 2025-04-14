// activation functions

use ndarray::{Array, Ix3};
use std::f32;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum ActivationFunctionType {
    None,
    Sigmoid,
    Relu,
    LeakyRelu,
    Tanh,
    Softmax,
}

pub fn sigmoid(x: &f32) -> f32 {
    let minus_x = -x;
    1.0 / (1.0 + minus_x.exp())
}

pub fn relu(x: &f32) -> f32 {
    x.max(0.0)
}

pub fn leaky_relu(x: &f32, alpha: Option<f32>) -> f32 {
    let leak = alpha.unwrap_or_else(|| 0.1) * x;
    x.max(leak)
}

pub fn tanh(x: &f32) -> f32 {
    x.clone().tanh()
}

pub fn softmax(x: &Array<f32, Ix3>) -> Array<f32, Ix3> {
    let exp_scores = x.clone().map(|x| x.exp());
    exp_scores.clone() / exp_scores.sum()
}
