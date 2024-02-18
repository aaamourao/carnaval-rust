use ndarray::{Array, Ix3};
use crate::activation::ActivationFunctionType;

pub struct Conv2D {
    pub filters: usize,
    pub kernel_size: usize,
    pub kernels: Array<f64, Ix3>,
    pub activation_function: ActivationFunctionType,
}