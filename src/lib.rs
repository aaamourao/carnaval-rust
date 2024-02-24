mod activation;
mod layer;
mod model;

#[cfg(test)]
mod tests {
    use more_asserts::{assert_ge, assert_le};
    use ndarray::{array};
    use plotpy::{Curve, Plot};
    use crate::activation::{ActivationFunctionType, relu, sigmoid};
    use crate::layer::conv2d::Conv2D;
    use crate::layer::Layer;
    use crate::layer::dense::Dense;
    use crate::model::Model;
    use crate::model::sequential::Sequential;

    #[test]
    fn relu_works() {
        let result = relu(&-2.0);
        let result_bigger_than_0 = relu(&3.0);
        assert_eq!(result, 0.0);
        assert_eq!(result_bigger_than_0, 3.0);
    }

    #[test]
    fn sigmoid_works() {
        let result = sigmoid(&0.0);
        let result_6 = sigmoid(&6.0);
        let result_minus_6 = sigmoid(&-6.0);
        assert_eq!(result, 0.5);
        assert_eq!(result_6, 0.9975273768433653);
        assert_eq!(result_minus_6, 0.0024726231566347743);
    }

    #[test]
    fn dense_works() {
        let nn = Dense::new(2
                            , 1, None);
        assert_eq!(nn.get_activation_function(), ActivationFunctionType::None);
        for weight in nn.weights.iter() {
            assert_le!(weight, &1.0_f64);
            assert_ge!(weight, &-1.0_f64);
        }
    }

    #[test]
    fn dense_forward() {
        let nn = Dense::new(2, 2, None);
        for weight in &nn.weights {
            println!("weight: {weight}");
        }
        let inference_result = &nn.forward(&array![[[1.], [1.]]]);

        if inference_result.is_ok() {
            for (row, value) in inference_result.as_ref().unwrap().iter().enumerate() {
                println!["value multiplied by 1.0 + bias({:?}): {value}", nn.bias[[0, row, 0]]]
            }
        }
    }

    #[test]
    fn dense_plot() {
        let nn = Dense::new(5, 5, Some(ActivationFunctionType::Relu));
        for weight in &nn.weights {
            println!("weight: {weight}");
        }

        let x = array![[[2.0], [4.0], [6.0], [8.0], [10.0]]];
        let y_result = &nn.forward(&x);
        let y = match y_result {
            Ok(result) => result,
            _ => panic!["Forward returned error"]
        };

        let mut curve = Curve::new();
        curve.set_label("y = nn.forward(x)");
        curve.draw(&x.into_raw_vec(), &y.clone().into_raw_vec());

        let mut plot = Plot::new();
        plot.set_title("NN with 5 neurons");
        plot.set_subplot(1, 1, 1)
            .set_title("Dense network test")
            .add(&curve)
            .grid_labels_legend("x", "y");

        let result = plot.save("plots/test.svg");
    }

    #[test]
    fn sequential_plot() {
        let layer0 = Dense::new(1, 1, Some(ActivationFunctionType::Relu));
        println!("layer0: {:?}", layer0.weights);
        let layer1 = Dense::new(1, 1, Some(ActivationFunctionType::Relu));
        println!("layer1: {:?}", layer1.weights);
        let layer2 = Dense::new(1, 1, Some(ActivationFunctionType::Relu));
        println!("layer1: {:?}", layer2.weights);

        let mut nn = Sequential::new(3);

        nn.push_layer("Input Layer".to_string(), Box::new(layer0));
        nn.push_layer("Hidden Layer 0".to_string(), Box::new(layer1));
        nn.push_layer("Hidden Layer 0".to_string(), Box::new(layer2));

        let x = array![[[-0.4], [-0.2], [0.0], [0.2], [0.4]]];
        let x = array![[[-10.0], [-7.5], [-5.0], [-2.5], [0.0], [2.5], [5.0], [7.5], [10.0]]];
        let y = x.map(|xx| nn.predict(&array![[[xx.clone()]]])[[0, 0, 0]]);

        let mut curve = Curve::new();
        curve.set_label("y = nn.predict(x)");
        curve.draw(&x.into_raw_vec(), &y.clone().into_raw_vec());

        let mut plot = Plot::new();
        plot.set_title("NN with 3 Dense layers");
        plot.set_subplot(1, 1, 1)
            .set_title("Sequential model test")
            .add(&curve)
            .grid_labels_legend("x", "y");

        let result = plot.save("plots/test_sequential.svg");
    }

    #[test]
    fn conv2d_basic_test() {
        let input = array![[
            [0., 1., 2., 3., 4., 5.],
            [5., 6., 7., 8., 9., 10.],
            [11., 12., 13., 14., 15., 16.],
            [17., 18., 19., 20., 21., 22.],
        ]];

        let input_shape = input.shape();

        let mut nn = Conv2D::new(1, 3, (input_shape[0], input_shape[1], input_shape[2]), None, None,
                                 None, None);

        let result = nn.forward(&input);
    }
}
