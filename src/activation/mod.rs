// activation functions

use std::f64;

pub fn sigm(x: f64) -> f64 {
    1 / (1 + x.exp())
}

pub fn relu(x: f64) -> f64 {
    x.max(0)
}