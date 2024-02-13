mod activation;
mod layer;
mod model;

#[cfg(test)]
mod tests {
    use std::ops::Mul;
    use ndarray::array;
    use crate::activation::{ActivationFunctionType, relu, sigmoid};
    use crate::layer::hidden_layer::HiddenLayer;
    use crate::layer::input_layer::InputLayer;
    use crate::layer::Layer;
    use crate::model::dense::Dense;

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
    fn dense_works() {
        let nn = Dense::new((1, 1, 2), 1);
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

    #[test]
    fn dense_forward() {
        let mut nn = Dense::new((1, 1, 3), 2);
        assert_eq!(nn.layers.len(), 2);
        nn.layers.last_mut().unwrap().initialize_weights_with_values(array![[
            [0.73, 0.2]
        ]]);
        let input= array![[[10.], [20.], [-20.], [-40.], [-3.]]];
        println!["{:?}", input.shape()];
        // let test0 = input.mul(array![])
        let test0 = array![[[1., 0., 0.]]];
        println!["{:?}", test0.shape()];
        let test1 = input.mul(test0);
        println!["{:?}", test1.shape()];
        //let result = model.forward(array![[[10.]], [[20.]], [[-20.]], [[-40.]], [[-3.]]]);

    }
}
