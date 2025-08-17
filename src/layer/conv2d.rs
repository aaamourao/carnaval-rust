//use rayon::iter::ParallelIterator;
use std::error::Error;
use std::ops::Mul;

use ndarray::{s, Array, Ix3, Zip};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

use crate::{
    activation::{leaky_relu, relu, sigmoid, tanh, ActivationFunctionType},
    layer::util::add_padding,
};

pub struct Conv2dLayer {
    filters: usize,
    kernel_size: usize,
    kernels: Vec<Array<f32, Ix3>>,
    padding: (usize, usize),
    input_dim: (usize, usize, usize),
    pub output_dim: (usize, usize, usize),
    strides: (usize, usize),
    dilatation_rate: (usize, usize),
    activation_function: ActivationFunctionType,
}

impl Conv2dLayer {
    pub fn new(
        filters: usize,
        kernel_size: usize,
        input_dim: (usize, usize, usize),
        padding: Option<(usize, usize)>,
        strides: Option<(usize, usize)>,
        dilation_rate: Option<(usize, usize)>,
        activation_function_type: Option<ActivationFunctionType>,
    ) -> Result<Self, Box<dyn Error>> {
        // TODO: kernel_size is not the only parameter that should be checked
        if kernel_size < 1 {
            return Err(Box::new(Conv2dError::KernelSizeError));
        }

        let padding = padding.unwrap_or((0, 0));
        let dilatation_rate = dilation_rate.unwrap_or((1, 1));
        let strides = strides.unwrap_or((1, 1));
        let activation_function = activation_function_type.unwrap_or(ActivationFunctionType::None);

        Ok(Self {
            filters,
            kernel_size,
            kernels: populate_kernels_with_random(kernel_size, filters, input_dim.2),
            padding,
            input_dim: (
                input_dim.0 + padding.0,
                input_dim.1 + padding.0,
                input_dim.2,
            ),
            output_dim: get_output_dim(
                input_dim,
                padding,
                kernel_size,
                dilatation_rate,
                strides,
                filters,
            ),
            strides,
            dilatation_rate,
            activation_function,
        })
    }

    pub fn activation_function(&self) -> ActivationFunctionType {
        self.activation_function
    }

    pub fn forward(&self, input: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>, Box<dyn Error>> {
        let input_padded = add_padding(input, &self.padding);
        let mut output = Array::zeros((self.output_dim.0, self.output_dim.1, self.output_dim.2));

        let input_padded_channel_size = input_padded.shape()[2];

        // TODO: Dilation and stride are not being considered now
        Zip::indexed(output.view_mut()).par_for_each(|(row, col, feature), value| {
            let max_row = row + self.kernel_size;
            let max_col = col + self.kernel_size;
            let kernel = &self.kernels[feature];

            let input_slice =
                input_padded.slice(s!(row..max_row, col..max_col, 0..input_padded_channel_size));

            /*let output_cel = Zip::from(kernel).and(&input_slice).par_fold(
                || 0.,
                |sum, kernel_elem, input_elem| sum + kernel_elem * input_elem,
                |sum, other_sum| sum + other_sum
            );*/
            let output_cel = kernel.mul(&input_slice).sum();

            *value = match self.activation_function {
                ActivationFunctionType::Relu => relu(&output_cel),
                ActivationFunctionType::Sigmoid => sigmoid(&output_cel),
                ActivationFunctionType::LeakyRelu => leaky_relu(&output_cel, None),
                ActivationFunctionType::Tanh => tanh(&output_cel),
                // TODO: enable softmax for Conv2D
                // ActivationFunctionType::Softmax => softmax(&output_celt),
                _ => output_cel,
            };
        });

        Ok(output)
    }
}

fn populate_kernels_with_random(
    kernel_size: usize,
    filters: usize,
    channels: usize,
) -> Vec<Array<f32, Ix3>> {
    let mut kernels = Vec::with_capacity(kernel_size);

    let mut i: usize = filters;
    while i >= 1 {
        let initial_filter = Array::random(
            (kernel_size, kernel_size, channels),
            Uniform::new(-1.0, 1.0),
        );
        kernels.push(initial_filter);
        i -= 1;
    }

    kernels
}

pub fn get_output_dim(
    input_dim: (usize, usize, usize),
    padding: (usize, usize),
    kernel_size: usize,
    dilatation_rate: (usize, usize),
    strides: (usize, usize),
    filters: usize,
) -> (usize, usize, usize) {
    let (input_height, input_width, _) = input_dim;
    let (padding_height, padding_width) = padding;
    let (dilatation_height, dilatation_width) = dilatation_rate;
    let (stride_height, stride_width) = strides;

    let height = 1
        + (input_height + 2 * padding_height
            - kernel_size
            - (kernel_size - 1) * (dilatation_height - 1))
            / stride_height;
    let width = 1
        + (input_width + 2 * padding_width
            - kernel_size
            - (kernel_size - 1) * (dilatation_width - 1))
            / stride_width;

    (height, width, filters)
}

#[derive(Debug, thiserror::Error)]
pub enum Conv2dError {
    #[error("invalid kernel size")]
    KernelSizeError,
}
