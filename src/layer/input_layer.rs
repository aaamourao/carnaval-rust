use ndarray::{Array, Ix3};
use crate::activation::ActivationFunctionType;
use crate::layer::{Layer, LayerType};

pub struct InputLayer {
    pub height: usize,
    pub width: usize,
    pub depth: usize,
    pub weights: Array<f64, Ix3>,
    activation_function: ActivationFunctionType,
}

impl InputLayer {
    pub fn new(depth: usize, height: usize,width: usize) -> Self {
        InputLayer {
            height,
            width,
            depth,
            weights: Array::ones((depth, height, width)),
            activation_function: ActivationFunctionType::None,
        }
    }
}
impl Layer for InputLayer {

    fn initialize_weights_with_random(&mut self) {
        // Do nothing... input layer does not modify the input
    }

    fn get_layer_type(&self) -> LayerType {
        LayerType::Input
    }

    fn get_activation_function(&self) -> ActivationFunctionType {
        self.activation_function
    }
}