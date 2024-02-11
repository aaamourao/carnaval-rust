use crate::nn::NeuralNetwork;

pub struct Conv {   
}

impl NeuralNetwork for Conv {
    fn new(layers_size: usize, input_dim: (usize, usize, usize)) -> Self {
        Conv {}
    }
}
