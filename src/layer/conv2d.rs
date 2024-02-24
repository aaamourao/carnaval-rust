use ndarray::{Array, ArrayViewMut, Axis, Ix3, s};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use crate::activation::ActivationFunctionType;
use crate::layer::{Layer, LayerError, LayerType};

pub struct Conv2D {
    filters: usize,
    kernel_size: usize,
    kernels: Vec<Array<f64, Ix3>>,
    padding: (usize, usize),
    input_dim: (usize, usize, usize),
    output_dim: (usize, usize, usize),
    strides: (usize, usize),
    dilatation_rate: (usize, usize),
    activation_function: ActivationFunctionType,
}

impl Conv2D {
    pub fn new(filters: usize,
               kernel_size: usize,
               input_dim: (usize, usize, usize),
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
            input_dim: (input_dim.0, input_dim.1 + padding.0, input_dim.2 + padding.1),
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
        let input_padded = add_padding(input, &self.padding);
        let (output_depth, output_height, output_width) = self.output_dim;
        let mut output = Array::zeros((self.filters * output_depth,
                                       output_height, output_width));

        let input_padded_shape = input_padded.shape();
        let input_padded_height = input_padded_shape[1];
        let input_padded_width = input_padded_shape[2];

        for kernel in self.kernels.iter() {
            for i in 0..input_padded_height - self.kernel_size {
                // TODO: for now, dilatation/stride is not being considered in the convolution
                let max_i = i + self.kernel_size;
                for j in 0..input_padded_width - self.kernel_size {
                    let max_j = j + self.kernel_size;
                    let input_slice = input_padded.slice(s![0..1, i..max_i, j..max_j]);
                    println!("{:?}", kernel.index_axis(Axis(0), 0).dot(&input_slice.index_axis(Axis(0), 0)));
                }
            }
        }

        Ok(output)
    }
}

fn populate_kernels_with_random(kernel_size: usize, filters: usize)
    -> Vec<Array<f64, Ix3>> {
    let mut kernels = Vec::with_capacity(kernel_size);

    let mut i: usize = filters;
    while i >= 1 {
        let initial_filter = Array::random((1, kernel_size, kernel_size),
                                           Uniform::new(-1.0, 1.0));
        kernels.push(initial_filter);
        i -= 1;
    }

    return kernels
}

fn add_padding(input: &Array<f64, Ix3>, padding: &(usize, usize)) -> Array<f64, Ix3> {
    let input_shape = input.shape();
    let input_depth = input_shape[0];
    let input_height = input_shape[1];
    let input_width = input_shape[2];


    let mut input_padded = Array::zeros((input_depth,
                                         input_height + padding.0, input_width + padding.1));

    input_padded.slice_mut(s![0..input_depth,
        padding.0..input_height, padding.1..input_width]).assign(input);

    return input_padded
}

fn get_output_dim(input_dim: (usize, usize, usize),
                  padding: (usize, usize),
                  kernel_size: usize,
                  dilatation_rate: (usize, usize),
                  strides: (usize, usize)) -> (usize, usize, usize) {
    let (input_depth, input_height, input_width) = input_dim;
    let (padding_height, padding_width) = padding;
    let (dilatation_height, dilatation_width) = dilatation_rate;
    let (stride_height, stride_width) = strides;

    let height = 1 + (input_height + 2 * padding_height - kernel_size - (kernel_size - 1)
        * (dilatation_height - 1)) / stride_height;
    let width = 1 + (input_width + 2 * padding_width - kernel_size - (kernel_size - 1)
        * (dilatation_width - 1)) / stride_width;

    (input_depth, height, width)
}