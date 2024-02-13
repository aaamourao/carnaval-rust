use ndarray::{Array, Ix3};

pub mod dense;

pub trait Model {
    fn forward(&self, input: Array<f64, Ix3>,) -> Array<f64, Ix3>;
}
