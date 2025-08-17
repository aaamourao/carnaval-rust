use std::error::Error;

use crate::activation::ActivationFunctionType;
use ndarray::{Array, Ix3};

#[derive(Debug, Default)]
pub struct FlattenLayer;

impl FlattenLayer {
    pub fn new() -> Self {
        Self
    }
}

impl FlattenLayer {
    pub fn activation_function(&self) -> ActivationFunctionType {
        ActivationFunctionType::None
    }

    pub fn forward(&self, input: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>, Box<dyn Error>> {
        let flatten_input = Array::from_iter(input.iter().copied());
        let flatten_input_size = flatten_input.shape()[0];
        Ok(flatten_input
            .to_shape((1, flatten_input_size, 1))
            .unwrap()
            .to_owned())
    }
}
