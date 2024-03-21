use ndarray::{Array, array, Axis, IntoDimension, Ix3};
use crate::activation::ActivationFunctionType;
use crate::layer::{Layer, LayerError, LayerType};

pub struct Flatten {
}

impl Flatten {
    pub fn new () -> Self {
        Flatten {}
    }
}

impl Layer for Flatten {
    fn get_layer_type(&self) -> LayerType {
        LayerType::Flatten
    }

    fn get_activation_function(&self) -> ActivationFunctionType {
        ActivationFunctionType::None
    }

    fn forward(&self, input: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>, LayerError> {
        let mut flatten_input = Array::from_iter(input.iter().cloned());
        let flatten_input_size = flatten_input.shape()[0];
        Ok(flatten_input.to_shape((1, flatten_input_size,  1)).unwrap().to_owned())
    }
}