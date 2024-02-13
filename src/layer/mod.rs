use crate::activation::ActivationFunctionType;

pub mod input_layer;
pub mod hidden_layer;

pub enum LayerType {
    Input,
    Hidden,
    Output
}

pub trait Layer {
    // For now, there is only one way of initializing weights
    fn initialize_weights_with_random(&mut self);

    fn get_layer_type(&self) -> LayerType;

    fn get_activation_function(&self) -> ActivationFunctionType;
}