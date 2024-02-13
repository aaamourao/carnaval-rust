mod activation;
mod layer;
mod nn;

use activation::{relu, sigmoid};

#[cfg(test)]
mod tests {
    use crate::layer::hidden_layer::HiddenLayer;
    use crate::layer::Layer;
    use super::*;

    #[test]
    fn relu_works() {
        let result = relu(-2.0);
        let result_bigger_than_0 = relu(3.0);
        assert_eq!(result, 0.0);
        assert_eq!(result_bigger_than_0, 3.0);
    }

    #[test]
    fn sigmoid_works() {
        let result = sigmoid(0.0);
        let result_6 = sigmoid(6.0);
        let result_minus_6 = sigmoid(-6.0);
        assert_eq!(result, 0.5);
        assert_eq!(result_6, 0.9975273768433653);
        assert_eq!(result_minus_6, 0.0024726231566347743);
    }

    #[test]
    fn hidden_layer_works() {
        let mut layer = HiddenLayer::new(1, 1, 2);
        layer.initialize_weights_with_random();
        println!["{:?}", layer.weights];
    }
}
