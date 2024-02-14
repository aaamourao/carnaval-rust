use std::ops::Mul;
use ndarray::{Array, Ix3};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use crate::activation::ActivationFunctionType;
use crate::layer::{Layer, LayerType};

pub struct Dense {
    pub nn_layers: Vec<Array<f64, Ix3>>,
    pub activation_function: ActivationFunctionType,
}

impl Dense {
    pub fn new(input_dim: (usize, usize, usize),
           output_height: usize,
           activation_function: Option<ActivationFunctionType>) -> Self {
        let (input_depth, input_height, input_width) = input_dim;

        let mut layers_size = if output_height < input_height {
            input_height - output_height
        } else {
            output_height - input_height
        };

        let mut dense_layers: Vec<Array<f64, Ix3>> = Vec::with_capacity(layers_size);

        // layers size is usize, so we need to increment it in order
        // to add the correct number of layers
        layers_size += 1;
        while layers_size > 0 {
            dense_layers.push(Array::random((input_depth, input_height, input_width),
                                            Uniform::new(0.0, 1.0)));
           layers_size -= 1;
        }

        Dense {
            nn_layers: dense_layers,
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
        let mut current = input;

        for layer in self.nn_layers.iter() {
            current = current.mul(layer);
        }

        return current
    }
}
