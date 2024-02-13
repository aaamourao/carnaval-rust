mod activation;
mod layer;
mod nn;

#[cfg(test)]
mod tests {
    use crate::activation::ActivationFunctionType;
    use crate::layer::hidden_layer::HiddenLayer;
    use crate::layer::input_layer::InputLayer;
    use crate::layer::Layer;
    use super::*;

    #[test]
    fn relu_works() {
        let result = activation::relu(-2.0);
        let result_bigger_than_0 = activation::relu(3.0);
        assert_eq!(result, 0.0);
        assert_eq!(result_bigger_than_0, 3.0);
    }

    #[test]
    fn sigmoid_works() {
        let result = activation::sigmoid(0.0);
        let result_6 = activation::sigmoid(6.0);
        let result_minus_6 = activation::sigmoid(-6.0);
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
        let conv = Conv::new((1, 1, 2), 1);
        for value in layer.weights.iter() {
            assert_eq!(value, &1.0_f64);
        }
    }
}
