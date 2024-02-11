// activation functions

use std::f64;

pub enum ActivationFunctionType {
    None,
    Sigmoid,
    Relu,
}

pub fn sigmoid(x: f64) -> f64 {
    let minus_x = -x;
    1.0 / (1.0 + minus_x.exp())
}

pub fn relu(x: f64) -> f64 {
    f64::max(x, 0.0)
}