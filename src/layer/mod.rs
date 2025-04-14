use std::error::Error;

use crate::activation::ActivationFunctionType;
use conv2d::Conv2dLayer;
use dense::DenseLayer;
use flatten::FlattenLayer;
use maxpool2d::MaxPool2dLayer;
use ndarray::{Array, Ix3};

pub mod conv2d;
pub mod dense;
pub mod flatten;
pub mod maxpool2d;
mod util;

pub enum Layer {
    Dense(DenseLayer),
    Conv2d(Conv2dLayer),
    MaxPool2d(MaxPool2dLayer),
    Flatten(FlattenLayer),
}

impl Layer {
    // For now, there is only one way of initializing weights
    // fn initialize_weights_with_values<T>(&mut self, values: T);

    pub fn activation_function(&self) -> ActivationFunctionType {
        match &self {
            Layer::Dense(dense) => dense.activation_function(),
            Layer::Conv2d(conv) => conv.activation_function(),
            Layer::MaxPool2d(max_pool) => max_pool.activation_function(),
            Layer::Flatten(flatten) => flatten.activation_function(),
        }
    }

    pub fn forward(&self, input: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>, Box<dyn Error>> {
        match &self {
            Layer::Dense(dense) => dense.forward(input),
            Layer::Conv2d(conv) => conv.forward(input),
            Layer::MaxPool2d(max_pool) => max_pool.forward(input),
            Layer::Flatten(flatten) => flatten.forward(input),
        }
    }
}
