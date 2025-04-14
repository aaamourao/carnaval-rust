#![expect(dead_code)]
pub mod activation;
pub mod layer;
pub mod model;

#[cfg(test)]
mod tests {
    use crate::activation::{relu, sigmoid, ActivationFunctionType};
    use crate::layer::conv2d::Conv2dLayer;
    use crate::layer::dense::DenseLayer;
    use crate::layer::flatten::FlattenLayer;
    use crate::layer::maxpool2d::MaxPool2dLayer;
    use crate::layer::Layer;
    use crate::model::sequential::SequentialModel;
    use more_asserts::{assert_ge, assert_le};
    use ndarray::array;
    //use ndarray_rand::RandomExt;
    use plotpy::{Curve, Plot};

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
        assert_eq!(result_6, 0.9975274);
        assert_eq!(result_minus_6, 0.002472623);
    }

    #[test]
    fn dense_works() {
        let nn = DenseLayer::new(2, 1, None);
        assert_eq!(nn.activation_function(), ActivationFunctionType::None);
        for weight in nn.weights.iter() {
            assert_le!(weight, &1.0_f32);
            assert_ge!(weight, &-1.0_f32);
        }
    }

    #[test]
    fn dense_forward() {
        let nn = DenseLayer::new(2, 2, None);
        for weight in &nn.weights {
            println!("weight: {weight}");
        }
        let inference_result = &nn.forward(&array![[[1.], [1.]]]);

        if inference_result.is_ok() {
            for (row, value) in inference_result.as_ref().unwrap().iter().enumerate() {
                println![
                    "value multiplied by 1.0 + bias({:?}): {value}",
                    nn.bias[[0, row, 0]]
                ]
            }
        }
    }

    #[test]
    fn dense_plot() {
        let nn = DenseLayer::new(5, 5, Some(ActivationFunctionType::Relu));
        for weight in &nn.weights {
            println!("weight: {weight}");
        }

        let x = array![[[2.0], [4.0], [6.0], [8.0], [10.0]]];
        let y_result = &nn.forward(&x);
        let y = match y_result {
            Ok(result) => result,
            _ => panic!["Forward returned error"],
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

        let _ = plot.save("plots/test.svg");
    }

    #[test]
    fn sequential_plot() {
        let layer0 = Layer::Dense(DenseLayer::new(1, 1, Some(ActivationFunctionType::Relu)));
        let layer1 = Layer::Dense(DenseLayer::new(1, 1, Some(ActivationFunctionType::Relu)));
        let layer2 = Layer::Dense(DenseLayer::new(1, 1, Some(ActivationFunctionType::Relu)));

        let mut nn = SequentialModel::new(3);

        nn.push_layer("Input Layer".to_string(), layer0);
        nn.push_layer("Hidden Layer 0".to_string(), layer1);
        nn.push_layer("Hidden Layer 0".to_string(), layer2);

        let x = array![[
            [-10.0],
            [-7.5],
            [-5.0],
            [-2.5],
            [0.0],
            [2.5],
            [5.0],
            [7.5],
            [10.0]
        ]];
        let y = x.map(|xx| nn.predict(&array![[[xx.clone()]]]).unwrap()[[0, 0, 0]]);

        let mut curve = Curve::new();
        curve.set_label("y = nn.predict(x)");
        curve.draw(&x.into_raw_vec(), &y.clone().into_raw_vec());

        let mut plot = Plot::new();
        plot.set_title("NN with 3 Dense layers");
        plot.set_subplot(1, 1, 1)
            .set_title("Sequential model test")
            .add(&curve)
            .grid_labels_legend("x", "y");

        let _ = plot.save("plots/test_sequential.svg");
    }

    #[test]
    fn conv2d_basic_test() {
        let input = array![
            [[0.], [1.], [2.], [3.], [4.], [5.]],
            [[6.], [7.], [8.], [9.], [10.], [11.]],
            [[12.], [13.], [14.], [15.], [16.], [17.]],
            [[18.], [19.], [20.], [21.], [22.], [23.]],
        ];

        let input_shape = input.shape();
        println!("{:?}", input_shape);

        let nn = Conv2dLayer::new(
            1,
            3,
            (input_shape[0], input_shape[1], input_shape[2]),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let result = nn.forward(&input).unwrap();
        println!("{:?}", result);
    }

    #[test]
    fn maxpool2d_basic_test() {
        let input = array![
            [[0.], [1.], [2.], [3.], [4.], [5.]],
            [[6.], [7.], [8.], [9.], [10.], [11.]],
            [[12.], [13.], [14.], [15.], [16.], [17.]],
            [[18.], [19.], [20.], [21.], [22.], [23.]],
        ];

        let nn = MaxPool2dLayer::new((2, 2), None, None);

        let result = nn.forward(&input).unwrap();
        assert_eq!(
            result,
            array![[[7.0], [9.0], [11.]], [[19.0], [21.0], [23.]]]
        )
    }

    #[test]
    fn flatten_basic_test() {
        let input = array![[
            [0., 1., 2., 3., 4., 5.],
            [6., 7., 8., 9., 10., 11.],
            [12., 13., 14., 15., 16., 17.],
            [18., 19., 20., 21., 22., 23.],
        ]];

        let nn = FlattenLayer::new();

        let result = nn.forward(&input).unwrap();

        assert_eq!(result.shape(), [1_usize, 24_usize, 1_usize]);
        assert_eq!(result[[0, 23, 0]], 23.);
    }
}
