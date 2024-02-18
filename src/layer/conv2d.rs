use ndarray::{Array, Ix3};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use crate::activation::ActivationFunctionType;

pub struct Conv2D {
    pub filter_dim: (usize, usize),
    pub kernel_size: usize,
    pub kernels: Vec<Array<f64, Ix3>>,
    pub is_padding_enabled: bool,
    pub strides: (usize, usize),
    pub dilatation_rate: (usize, usize),
    pub activation_function: ActivationFunctionType,
}

impl Conv2D {
    pub fn new(filter_dim: (usize, usize),
               kernel_size: usize,
               is_padding_enabled: Option<bool>,
               strides: Option<(usize, usize)>,
               dilation_rate: Option<(usize, usize)>,
               activation_function_type: Option<ActivationFunctionType>) -> Self {
        if kernel_size < 1 {
            panic!["kernel_size should be at least 1"];
        }

        Conv2D {
            filter_dim,
            kernel_size,
            is_padding_enabled: is_padding_enabled.unwrap_or(true),
            kernels: populate_kernels_with_random(kernel_size, filter_dim),
            strides: strides.unwrap_or((1, 1)),
            dilatation_rate: dilation_rate.unwrap_or((1, 1)),
            activation_function: activation_function_type.unwrap_or(ActivationFunctionType::None),
        }
    }
}

fn populate_kernels_with_random(kernel_size: usize, filter_dim: (usize, usize))
    -> Vec<Array<f64, Ix3>> {
    let mut kernels = Vec::with_capacity(kernel_size);
    let (filter_height, filter_width) = filter_dim;

    let initial_filter = Array::random((1, filter_height, filter_width),
                                       Uniform::new(-1.0, 1.0));

    let mut i: usize = kernel_size;
    while i > 1 {
        kernels.push(initial_filter.clone());
    }

    return kernels
}