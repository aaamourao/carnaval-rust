use ndarray::{Array, Array3, ArrayView, Axis, Ix2, Ix3};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use crate::activation::{ActivationFunctionType, relu, sigmoid};
use crate::layer::{ForwardError, Layer, LayerType};

pub struct Dense {
    pub input_size: usize,
    pub output_size: usize,
    pub layers: Array<f64, Ix3>,
    pub bias: Array<f64, Ix3>,
    pub activation_function: ActivationFunctionType,
}

impl Dense {
    pub fn new(input_size: usize,
               output_size: usize,
               activation_function: Option<ActivationFunctionType>) -> Self {

        let layers = Array::random((1, output_size, input_size), Uniform::new(0.0, 1.0));
        let bias = Array::random((1, output_size, 1), Uniform::new(0.0, 1.0));

        Dense {
            input_size,
            output_size,
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

    fn forward(&self, input: Array<f64, Ix3>) -> Result<Array<f64, Ix3>, ForwardError> {
        let result = if input.len_of(Axis(0)) != 1 && input.len_of(Axis(2)) != 1 {
            Err(ForwardError::IncorrectDimensions(
                "input should have dimensions (1, input_size, 1)".to_string()
            ))
        } else {
            let mut current_result = Array3::<f64>::zeros((1, self.output_size, 1));

            let layer_closure = |layer: &ArrayView<f64, Ix2>, row: usize| -> f64 {
                //(layer * input.slice(s![0, .., ..])) + self.bias[[0, row, 0]]
                ((layer * &input.index_axis(Axis(0), 0)) + self.bias[[0, row, 0]])[[0, 0]]
            };

            let mut row: usize = 0;

            for layer in self.layers.axis_iter(Axis(1)) {
                current_result[[0, row, 0]] = match self.activation_function {
                    ActivationFunctionType::Relu => relu(layer_closure(&layer, row)),
                    ActivationFunctionType::Sigmoid => sigmoid(layer_closure(&layer, row)),
                    _ => layer_closure(&layer, row),
                };
                row += 1;
            }
            Ok(current_result)
        };
        return result
    }
}
