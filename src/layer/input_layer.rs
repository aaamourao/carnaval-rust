use crate::activation::ActivationFunctionType;
use crate::layer::{Layer, LayerType};

struct InputLayer {
    length: usize,
    weights: Vec<f64>,
    activation_function: ActivationFunctionType,
}
impl Layer for InputLayer {
    fn new(length: usize) -> Self {
        InputLayer {
            length,
            weights: vec![1.0, length],
            activation_function: ActivationFunctionType::None,
        }
    }

    fn initialize_weights() {
        // Do nothing... input layer does not modify the input
    }

    fn get_layer_type() -> LayerType {
        LayerType::Input
    }
}