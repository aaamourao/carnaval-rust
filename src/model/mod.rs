pub mod sequential;

use ndarray::{Array, Ix2, Ix3};

pub trait Model {
    fn predict(&self, input: &Array<f64, Ix3>) -> Array<f64, Ix3>;

    fn forward(&self, input: &Array<f64, Ix3>, target: Array<f64, Ix2>) -> Array<f64, Ix3>;
}
