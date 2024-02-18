use ndarray::{Array, Ix3};
use crate::activation::ActivationFunctionType;

pub mod dense;

pub enum LayerType {
    Dense,
}

#[derive(Debug)]
pub enum ForwardError {
    IncorrectDimensions(String),
}

pub trait Layer {
    // For now, there is only one way of initializing weights
    // fn initialize_weights_with_values<T>(&mut self, values: T);

    fn get_layer_type(&self) -> LayerType;

    fn get_activation_function(&self) -> ActivationFunctionType;

    fn forward(&self, input: &Array<f64, Ix3>,) -> Result<Array<f64, Ix3>, ForwardError>;
}
