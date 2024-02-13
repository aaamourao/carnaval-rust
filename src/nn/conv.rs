use ndarray::{Array, Ix3};
use crate::layer::hidden_layer::HiddenLayer;
use crate::layer::input_layer::InputLayer;
use crate::layer::Layer;
use crate::nn::NeuralNetwork;

pub struct Conv {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Conv {
    pub fn new(input_dim: (usize, usize, usize), output_height: usize) -> Self {
        let (mut input_depth, input_height, input_width) = input_dim;

        let layers_size = if output_height < input_height {
            input_height - output_height
        } else {
            output_height - input_height
        };

        let mut conv_layers: Vec<Box<dyn Layer>> = Vec::with_capacity(layers_size);
        conv_layers.push(Box::new(InputLayer::new(input_depth, input_height, input_width)));

        while input_depth > 0 {
            input_depth -= 1;
            conv_layers.push(Box::new(HiddenLayer::new(input_depth, input_height, input_width)));
        }

        Conv {
            layers: conv_layers,
        }
    }
}

impl NeuralNetwork for Conv {
    fn forward(input: Array<f64, Ix3>) -> Array<f64, Ix3> {
        println!["{:?}", input];
        input
    }
}
