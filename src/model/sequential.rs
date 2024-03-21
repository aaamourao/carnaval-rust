use std::time::Instant;
use ndarray::{Array, Ix3};
use crate::layer::{Layer};
use crate::model::Model;

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    layer_names: Vec<String>,
}

impl Sequential {
    pub fn new(layers_size: usize) -> Self {
        Sequential {
            layers: Vec::with_capacity(layers_size),
            layer_names: Vec::with_capacity(layers_size),
        }
    }

    pub fn push_layer(&mut self, layer_name: String, layer: Box<dyn Layer>) {
        self.layers.push(layer);
        self.layer_names.push(layer_name);
    }
}

impl Model for Sequential {
    fn predict(&self, input: &Array<f32, Ix3>) -> Array<f32, Ix3> {
        let mut current = input.clone();

        for (index, layer) in self.layers.iter().enumerate() {
            let result = layer.forward(&current);
            current = current + match result {
                Ok(forward_result) => forward_result,
                Err(err) => panic!("Predict Error, Layer {:?}: {:?}", self.layer_names[index], err),
            };
        }

        return current
    }

    fn forward(&self, input: &Array<f32, Ix3>) -> Array<f32, Ix3> {
        let mut result = input.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            let start = Instant::now();
            result = layer.forward(&result).unwrap();
            let duration = start.elapsed();
            println!("{} layer spent {:?}", self.layer_names[index], duration);

        }
        return result
    }
}