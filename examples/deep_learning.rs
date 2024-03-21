use std::time::Instant;
use ndarray::{Array, Ix3};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use carnaval_rust::activation::ActivationFunctionType;
use carnaval_rust::layer::conv2d::Conv2D;
use carnaval_rust::layer::dense::Dense;
use carnaval_rust::layer::flatten::Flatten;
use carnaval_rust::layer::maxpool2d::MaxPool2D;
use carnaval_rust::model::Model;
use carnaval_rust::model::sequential::Sequential;

fn create_deep_learning_model(input_shape: &[usize]) -> Sequential {

    let layer0 = Conv2D::new(32, 3, (input_shape[0], input_shape[1], input_shape[2]), None, None,
                             None, Some(ActivationFunctionType::Relu));
    let layer1 = MaxPool2D::new((2, 2), None, None);

    let layer2 = Conv2D::new(32, 3, layer1.get_output_dim(layer0.output_dim), None, None,
                             None, Some(ActivationFunctionType::Relu));
    let layer3 = MaxPool2D::new((2, 2), None, None);

    let layer4 = Conv2D::new(64, 3, layer3.get_output_dim(layer2.output_dim), None, None,
                             None, Some(ActivationFunctionType::Relu));
    let layer5 = MaxPool2D::new((2, 2), None, None);

    let shape = layer5.get_output_dim(layer4.output_dim);
    let dense_0_input_size = shape.0 * shape.1 * shape.2;

    let layer6 = Flatten::new();
    let layer7 = Dense::new(dense_0_input_size, 128, Some(ActivationFunctionType::Relu));
    let layer8 = Dense::new(128, 2, Some(ActivationFunctionType::Sigmoid));

    let mut nn = Sequential::new(9);

    nn.push_layer("Conv2D Layer 0".to_string(), Box::new(layer0));
    nn.push_layer("MaxPool2D Layer 0".to_string(), Box::new(layer1));
    nn.push_layer("Conv2D Layer 1".to_string(), Box::new(layer2));
    nn.push_layer("MaxPool2D Layer 1".to_string(), Box::new(layer3));
    nn.push_layer("Conv2D Layer 2".to_string(), Box::new(layer4));
    nn.push_layer("MaxPool2D Layer 2".to_string(), Box::new(layer5));
    nn.push_layer("Flatten Layer".to_string(), Box::new(layer6));
    nn.push_layer("Dense layer 0".to_string(), Box::new(layer7));
    nn.push_layer("Dense layer 1 (classification)".to_string(), Box::new(layer8));

    nn
}

fn main() {
    let input = Array::random((224, 224, 3),
                              Uniform::new(0., 1.));
    let model = create_deep_learning_model(input.shape());
    let start_time = Instant::now();
    let result = model.forward(&input);
    let duration = Instant::now() - start_time;
    println!("CNN inference time spent {:?}", duration);
    println!("Result: {:?}", result);
}