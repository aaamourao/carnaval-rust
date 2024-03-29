pub mod sequential;

use ndarray::{Array, Ix3};

pub trait Model {
    fn predict(&self, input: &Array<f32, Ix3>) -> Array<f32, Ix3>;

    fn forward(&self, input: &Array<f32, Ix3>) -> Array<f32, Ix3>;
}
