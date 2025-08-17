use std::error::Error;
use std::fmt;

use crate::activation::{leaky_relu, relu, sigmoid, softmax, tanh, ActivationFunctionType};
use ndarray::{Array, Axis, Ix3};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
//use rayon::iter::ParallelIterator;

pub struct DenseLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Array<f32, Ix3>,
    pub bias: Array<f32, Ix3>,
    pub activation_function: ActivationFunctionType,
}

/*
 * Dense handles 2D data, but its input is Ix3 arrays in order to be compatible with other layers
 * types
 */
impl DenseLayer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation_function: Option<ActivationFunctionType>,
    ) -> Self {
        let layers = Array::random((input_size, output_size, 1), Uniform::new(-1.0, 1.0));
        let bias = Array::random((1, output_size, 1), Uniform::new(-10.0, 10.0));

        DenseLayer {
            input_size,
            output_size,
            weights: layers,
            bias,
            activation_function: activation_function.unwrap_or(ActivationFunctionType::None),
        }
    }
}

impl DenseLayer {
    pub fn activation_function(&self) -> ActivationFunctionType {
        self.activation_function
    }

    pub fn forward(&self, input: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>, Box<dyn Error>> {
        if input.len_of(Axis(0)) != 1 && input.len_of(Axis(2)) != 1 {
            return Err(Box::new(InvalidDimensionsError));
        }
            let partial_result = &input
                .index_axis(Axis(2), 0)
                .dot(&self.weights.index_axis(Axis(2), 0))
                .to_shape((1, self.output_size, 1))?
                + &self.bias;

            let result = match self.activation_function {
                ActivationFunctionType::Relu => partial_result.map(relu),
                ActivationFunctionType::Sigmoid => partial_result.map(sigmoid),
                ActivationFunctionType::LeakyRelu => partial_result.map(|x| leaky_relu(x, None)),
                ActivationFunctionType::Tanh => partial_result.map(tanh),
                ActivationFunctionType::Softmax => softmax(&partial_result),
                ActivationFunctionType::None => partial_result.clone(),
            };
            Ok(result)

    }
}

#[derive(Debug)]
pub struct InvalidDimensionsError;

impl Error for InvalidDimensionsError {}

impl fmt::Display for InvalidDimensionsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Dimensions should match for forwarding")
    }
}
