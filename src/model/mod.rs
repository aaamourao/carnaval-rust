pub mod sequential;

use std::error::Error;

use ndarray::{Array, Ix3};
use sequential::SequentialModel;

pub enum Model {
    Sequential(SequentialModel),
}

impl Model {
    fn predict(&self, input: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>, Box<dyn Error>> {
        match &self {
            Model::Sequential(sequential) => sequential.predict(input),
        }
    }

    fn forward(&self, input: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>, Box<dyn Error>> {
        match &self {
            Model::Sequential(sequential) => sequential.predict(input),
        }
    }
}
