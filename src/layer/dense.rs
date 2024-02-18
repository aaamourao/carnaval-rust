use ndarray::{Array, Axis, Ix3};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use crate::activation::{ActivationFunctionType, leaky_relu, relu, sigmoid, softmax, tanh};
use crate::layer::{LayerError, Layer, LayerType};

pub struct Dense {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Array<f64, Ix3>,
    pub bias: Array<f64, Ix3>,
    pub activation_function: ActivationFunctionType,
}

/*
 * Dense handles 2D data, but its input is Ix3 arrays in order to be compatible with other layers
 * types
 */
impl Dense {
    pub fn new(input_size: usize,
               output_size: usize,
               activation_function: Option<ActivationFunctionType>) -> Self {

        let layers = Array::random((1, output_size, input_size),
                                   Uniform::new(0.0, 1.0));
        let bias = Array::random((1, output_size, 1),
                                 Uniform::new(0.0, 1.0));

        Dense {
            input_size,
            output_size,
            weights: layers,
            bias,
            activation_function: activation_function.unwrap_or(ActivationFunctionType::None),
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

    fn forward(&self, input: &Array<f64, Ix3>) -> Result<Array<f64, Ix3>, LayerError> {
        let result = if input.len_of(Axis(0)) != 1 && input.len_of(Axis(2)) != 1 {
            Err(LayerError::IncorrectDimensions(
                "input should have dimensions (1, input_size, 1)".to_string()
            ))
        } else {

            let partial_result = &self.weights.index_axis(Axis(0), 0).dot(
                &input.index_axis(Axis(0), 0)) + &self.bias;

            let result = match self.activation_function {
                ActivationFunctionType::Relu => partial_result.map(|x| relu(x)),
                ActivationFunctionType::Sigmoid => partial_result.map(|x| sigmoid(x)),
                ActivationFunctionType::LeakyRelu => partial_result.map(|x| leaky_relu(x, None)),
                ActivationFunctionType::Tanh => partial_result.map(|x| tanh(x)),
                ActivationFunctionType::Softmax => softmax(&partial_result),
                ActivationFunctionType::None => partial_result,
            };
            Ok(result)
        };
        return result
    }
}
