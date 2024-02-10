// activation functions

use std::f64;

pub fn sigm(x: f64) -> f64 {
    let minus_x = -x;
    1_f64 / (1_f64 + minus_x.exp())
}

pub fn relu(x: f64) -> f64 {
    x.max(0.0_f64)
}