use std::ops::{Add, Mul};
use ndarray::{Array, Ix3};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use crate::activation::ActivationFunctionType;
use crate::layer::{Layer, LayerType};

pub struct Dense {
    pub layers: Array<f64, Ix3>,
    pub bias: Array<f64, Ix3>,
    pub activation_function: ActivationFunctionType,
}

impl Dense {
    pub fn new(input_dim: usize,
           output_dim: usize,
           activation_function: Option<ActivationFunctionType>) -> Self {

        let layers = Array::random((1, input_dim, output_dim), Uniform::new(0.0, 1.0));
        let bias = Array::random((1, 1, output_dim), Uniform::new(0.0, 1.0));

        Dense {
            layers,
            bias,
            activation_function: activation_function.unwrap_or(ActivationFunctionType::Relu),
        }
    }
}

impl Layer for Dense {
    fn get_layer_type(&self) -> LayerType {
        LayerType::Dense
    }

    fn get_activation_function(&self) -> ActivationFunctionType {
        return self.activation_function
    }

    fn forward(&self, input: Array<f64, Ix3>) -> Array<f64, Ix3> {
        (&self.layers * &input) + &self.bias
    }
}
