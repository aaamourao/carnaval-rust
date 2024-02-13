use ndarray::{Array, Ix3};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use crate::activation::ActivationFunctionType;
use crate::layer::{Layer, LayerType};

pub struct HiddenLayer {
    height: usize,
    width: usize,
    depth: usize,
    pub weights: Array<f64, Ix3>,
    activation_function: ActivationFunctionType,
}

impl HiddenLayer {
    pub fn new(depth: usize, height: usize, width: usize) -> Self {
        HiddenLayer {
            height,
            width,
            depth,
            weights: Array::ones((depth, height, width)),
            activation_function: ActivationFunctionType::Relu
        }
    }
}

impl Layer for HiddenLayer {
    fn initialize_weights_with_random(&mut self) {
        self.weights = Array::random((self.depth, self.height, self.width),
                                     Uniform::new(0.0, 1.0));
    }

    fn initialize_weights_with_values(&mut self, values: Array<f64, Ix3>) {
        self.weights = values;
    }
    fn get_layer_type(&self) -> LayerType {
        LayerType::Hidden
    }

    fn get_activation_function(&self) -> ActivationFunctionType {
        self.activation_function
    }

    fn get_weights(&self) -> &Array<f64, Ix3> {
        &self.weights
    }
}