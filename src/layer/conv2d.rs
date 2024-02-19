use ndarray::{Array, Ix3};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use crate::activation::ActivationFunctionType;
use crate::layer::{Layer, LayerError, LayerType};

pub struct Conv2D {
    filters: usize,
    kernel_size: usize,
    kernels: Vec<Array<f64, Ix3>>,
    padding: (usize, usize),
    input_dim: (usize, usize),
    output_dim: (usize, usize),
    strides: (usize, usize),
    dilatation_rate: (usize, usize),
    activation_function: ActivationFunctionType,
}

impl Conv2D {
    pub fn new(filters: usize,
               kernel_size: usize,
               input_dim: (usize, usize),
               padding: Option<(usize, usize)>,
               strides: Option<(usize, usize)>,
               dilation_rate: Option<(usize, usize)>,
               activation_function_type: Option<ActivationFunctionType>) -> Self {
        // TODO: kernel_size is not the only parameter that should be checked
        if kernel_size < 1 {
            panic!["kernel_size should be at least 1"];
        }

        let padding = padding.unwrap_or((0, 0));
        let dilatation_rate = dilation_rate.unwrap_or((1, 1));
        let strides = strides.unwrap_or((1, 1));
        let activation_function =
            activation_function_type.unwrap_or(ActivationFunctionType::None);

        Conv2D {
            filters,
            kernel_size,
            kernels: populate_kernels_with_random(kernel_size, filters),
            padding,
            input_dim: (input_dim.0 + padding.0, input_dim.1 + padding.1),
            output_dim: get_output_dim(input_dim, padding, kernel_size, dilatation_rate, strides),
            strides,
            dilatation_rate,
            activation_function,
        }
    }
}

impl Layer for Conv2D {
    fn get_layer_type(&self) -> LayerType {
        LayerType::Conv2D
    }

    fn get_activation_function(&self) -> ActivationFunctionType {
        self.activation_function
    }

    fn forward(&self, input: &Array<f64, Ix3>) -> Result<Array<f64, Ix3>, LayerError> {
        let (output_height, output_width) = self.output_dim;
        let mut output = Array::zeros((self.filters,
                                       output_height, output_width));

        Ok(output)
    }
}

fn populate_kernels_with_random(kernel_size: usize, filters: usize)
    -> Vec<Array<f64, Ix3>> {
    let mut kernels = Vec::with_capacity(kernel_size);

    let initial_filter = Array::random((1, kernel_size, kernel_size),
                                       Uniform::new(-1.0, 1.0));

    let mut i: usize = filters;
    while i > 1 {
        kernels.push(initial_filter.clone());
        i -= 1;
    }

    return kernels
}

fn add_padding(input: &mut Array<f64, Ix3>, kernel_size: &usize) -> Array<f64, Ix3> {
    todo!()
}

fn get_output_dim(input_dim: (usize, usize),
                  padding: (usize, usize),
                  kernel_size: usize,
                  dilatation_rate: (usize, usize),
                  strides: (usize, usize)) -> (usize, usize) {
    let (input_height, input_width) = input_dim;
    let (padding_height, padding_width) = padding;
    let (dilatation_height, dilatation_width) = dilatation_rate;
    let (stride_height, stride_width) = strides;

    let height = 1 + (input_height + 2 * padding_height - kernel_size - (kernel_size - 1)
        * (dilatation_height - 1)) / stride_height;
    let width = 1 + (input_width + 2 * padding_width - kernel_size - (kernel_size - 1)
        * (dilatation_width - 1)) / stride_width;

    (height, width)
}