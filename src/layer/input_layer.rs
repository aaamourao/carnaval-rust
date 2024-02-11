use ndarray::{Array, Ix3};
use crate::activation::ActivationFunctionType;
use crate::layer::{Layer, LayerType};

pub struct InputLayer {
    height: usize,
    width: usize,
    depth: usize,
    weights: Array::<f64, Ix3>,
    activation_function: ActivationFunctionType,
}
impl Layer for InputLayer {
    fn new(depth: usize, height: usize,width: usize) -> Self {
        InputLayer {
            height,
            width,
            depth,
            weights: Array::ones((depth, height, width)),
            activation_function: ActivationFunctionType::None,
        }
    }

    fn initialize_weights_with_random(&self) {
        // Do nothing... input layer does not modify the input
    }

    fn get_layer_type(&self) -> LayerType {
        LayerType::Input
    }

    fn get_activation_function(&self) -> ActivationFunctionType {
        self.activation_function
    }
}