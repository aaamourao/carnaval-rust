use crate::layer::Layer;

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new(layers_size: usize) -> Self {
        Sequential {
            layers: Vec::with_capacity(layers_size),
        }
    }

    fn push_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }
}