use ndarray::{Array, Ix3};
use crate::activation::ActivationFunctionType;

pub mod dense;
pub mod conv2d;
pub mod maxpool2d;
mod util;
pub mod flatten;

pub enum LayerType {
    Dense,
    Conv2D,
    MaxPool,
    Flatten,
    Dropout,
}

#[derive(Debug)]
pub enum LayerError {
    IncorrectDimensions(String),
}

pub trait Layer {
    // For now, there is only one way of initializing weights
    // fn initialize_weights_with_values<T>(&mut self, values: T);

    fn get_layer_type(&self) -> LayerType;

    fn get_activation_function(&self) -> ActivationFunctionType;

    fn forward(&self, input: &Array<f32, Ix3>,) -> Result<Array<f32, Ix3>, LayerError>;
}
