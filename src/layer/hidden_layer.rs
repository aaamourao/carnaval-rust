use ndarray::{Array, Ix3};
use crate::activation::ActivationFunctionType;
use crate::layer::{Layer, LayerType};

pub struct HiddenLayer {
    height: usize,
    width: usize,
    depth: usize,
    weights: Array::<f64, Ix3>,
    activation_function: ActivationFunctionType,
}

impl Layer for HiddenLayer {
    fn new(depth: usize, height: usize, width: usize) -> Self {
        HiddenLayer {
            height,
            width,
            depth,
            weights: Array::ones((depth, height, width)),
            activation_function: ActivationFunctionType::Relu
        }
    }

    fn initialize_weights_with_random() {

    }

    fn get_layer_type() -> LayerType {
        LayerType::Hidden
    }
}