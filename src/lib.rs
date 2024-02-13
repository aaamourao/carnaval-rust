mod activation;
mod layer;
mod nn;

#[cfg(test)]
mod tests {
    use crate::activation::{ActivationFunctionType, relu, sigmoid};
    use crate::layer::hidden_layer::HiddenLayer;
    use crate::layer::input_layer::InputLayer;
    use crate::layer::Layer;
    use crate::nn::conv::Conv;

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

    #[test]
    fn hidden_layer_default_activation_function_check() {
        let mut layer = HiddenLayer::new(1, 1, 2);
        assert_eq!(layer.get_activation_function(), ActivationFunctionType::Relu);
    }

    #[test]
    fn input_layer_works() {
        let mut layer = InputLayer::new(1, 1, 2);
        layer.initialize_weights_with_random(); // does nothing
        for value in layer.weights.iter() {
            assert_eq!(value, &1.0_f64);
        }
    }

    #[test]
    fn input_layer_default_activation_function_check() {
        let mut layer = InputLayer::new(1, 1, 2);
        assert_eq!(layer.get_activation_function(), ActivationFunctionType::None);
    }

    #[test]
    fn conv_works() {
        let nn = Conv::new((1, 1, 2), 1);
        for (i, layer) in nn.layers.iter().enumerate() {
            if i == 0 {
                assert_eq!(layer.get_activation_function(), ActivationFunctionType::None);
                let weights = layer.get_weights();
                for value in weights.iter() {
                    assert_eq!(value, &1.0_f64);
                }
            } else {
                assert_eq!(layer.get_activation_function(), ActivationFunctionType::Relu);
            }
        }
    }
}
