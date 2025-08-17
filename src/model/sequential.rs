use std::error::Error;

use crate::layer::Layer;
use ndarray::{Array, Ix3};

pub struct SequentialModel {
    layers: Vec<Layer>,
    layer_names: Vec<String>,
}

impl SequentialModel {
    pub fn new(layers_size: usize) -> Self {
        SequentialModel {
            layers: Vec::with_capacity(layers_size),
            layer_names: Vec::with_capacity(layers_size),
        }
    }

    pub fn push_layer(&mut self, layer_name: String, layer: Layer) {
        self.layers.push(layer);
        self.layer_names.push(layer_name);
    }
}

impl SequentialModel {
    pub fn predict(&self, input: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>, Box<dyn Error>> {
        let mut current = input.clone();

        for layer in self.layers.iter() {
            let result = match layer.forward(&current) {
                Ok(forward_result) => forward_result,
                Err(err) => {
                    return Err(err);
                }
            };
            current = current + result;
        }

        Ok(current)
    }

    pub fn forward(&self, input: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>, Box<dyn Error>> {
        let mut result = input.clone();
        for layer in self.layers.iter() {
            result = layer.forward(&result)?;
        }
        Ok(result)
    }
}
