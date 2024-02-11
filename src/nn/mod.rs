mod conv;

pub trait NeuralNetwork {
    fn new(layers_size: usize, input_dim: (usize, usize, usize)) -> Self;
}
