use crate::activation::ActivationFunctionType;
use crate::layer::{Layer, LayerType};

struct HiddenLayer {
    length: usize,
    weights: Vec<f64>,
    activation_function: ActivationFunctionType,
}

impl Layer for HiddenLayer {
    fn new(length: usize) -> Self {
        HiddenLayer {
            length,
            weights: vec![1.0, length],
            activation_function: ActivationFunctionType::Relu
        }
    }

    fn initialize_weights() {
        todo!()
    }

    fn get_layer_type() -> LayerType {
        LayerType::Hidden
    }
}