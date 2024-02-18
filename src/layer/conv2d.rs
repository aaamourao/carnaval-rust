use ndarray::{Array, Ix3};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use crate::activation::ActivationFunctionType;
use crate::layer::{Layer, LayerError, LayerType};

pub struct Conv2D {
    pub filters: usize,
    pub kernel_size: usize,
    pub kernels: Vec<Array<f64, Ix3>>,
    pub is_padding_enabled: bool,
    pub strides: (usize, usize),
    pub dilatation_rate: (usize, usize),
    pub activation_function: ActivationFunctionType,
}

impl Conv2D {
    pub fn new(filters: usize,
               kernel_size: usize,
               is_padding_enabled: Option<bool>,
               strides: Option<(usize, usize)>,
               dilation_rate: Option<(usize, usize)>,
               activation_function_type: Option<ActivationFunctionType>) -> Self {
        // TODO: kernel_size is not the only parameter that should be checked
        if kernel_size < 1 {
            panic!["kernel_size should be at least 1"];
        }

        Conv2D {
            filters,
            kernel_size,
            is_padding_enabled: is_padding_enabled.unwrap_or(true),
            kernels: populate_kernels_with_random(kernel_size, filters),
            strides: strides.unwrap_or((1, 1)),
            dilatation_rate: dilation_rate.unwrap_or((1, 1)),
            activation_function: activation_function_type.unwrap_or(ActivationFunctionType::None),
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
        todo!()
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