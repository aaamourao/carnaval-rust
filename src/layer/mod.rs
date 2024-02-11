use crate::activation::ActivationFunctionType;

pub mod input_layer;
mod hidden_layer;

enum LayerType {
    Input,
    Hidden,
    Output
}

trait Layer {
    fn new(length: usize) -> Self;

    // For now, there is only one way of initializing weights
    fn initialize_weights();

    fn get_layer_type() -> LayerType;
}