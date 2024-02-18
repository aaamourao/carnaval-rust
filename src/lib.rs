mod activation;
mod layer;
mod model;

#[cfg(test)]
mod tests {
    use more_asserts::{assert_ge, assert_le};
    use ndarray::{array};
    use plotpy::{Curve, Plot};
    use crate::activation::{ActivationFunctionType, relu, sigmoid};
    use crate::layer::Layer;
    use crate::layer::dense::Dense;

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
    fn dense_works() {
        let nn = Dense::new(2
                            , 1, None);
        assert_eq!(nn.get_activation_function(), ActivationFunctionType::None);
        for weight in nn.layers.iter() {
            assert_le!(weight, &1.0_f64);
            assert_ge!(weight, &0.0_f64);
        }
    }

    #[test]
    fn dense_forward() {
        let nn = Dense::new(2, 2, None);
        for weight in &nn.layers {
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
        let nn = Dense::new(5, 5, Some(ActivationFunctionType::Sigmoid));
        for weight in &nn.layers {
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
        plot.set_title("NN with 5 hidden layers");
        plot.set_subplot(1, 1, 1)
            .set_title("Dense network test")
            .add(&curve)
            .grid_labels_legend("x", "y");

        let result = plot.save("plots/test.svg");
    }
}
