use std::time::SystemTime;

use carnaval_rust::{
    activation::ActivationFunctionType,
    layer::{
        conv2d::Conv2dLayer, dense::DenseLayer, flatten::FlattenLayer, maxpool2d::MaxPool2dLayer,
        Layer,
    },
    model::sequential::SequentialModel,
};
use ndarray::Array;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

fn create_deep_learning_model(input_shape: &[usize]) -> SequentialModel {
    let layer0 = Conv2dLayer::new(
        32,
        3,
        (input_shape[0], input_shape[1], input_shape[2]),
        None,
        None,
        None,
        Some(ActivationFunctionType::Relu),
    )
    .expect("Kernel size should be ok here");

    let layer1 = MaxPool2dLayer::new((2, 2), None, None);

    let layer2 = Conv2dLayer::new(
        32,
        3,
        layer1.output_dim(layer0.output_dim),
        None,
        None,
        None,
        Some(ActivationFunctionType::Relu),
    )
    .expect("Kernel size should be ok here");

    let layer3 = MaxPool2dLayer::new((2, 2), None, None);

    let layer4 = Conv2dLayer::new(
        64,
        3,
        layer3.output_dim(layer2.output_dim),
        None,
        None,
        None,
        Some(ActivationFunctionType::Relu),
    )
    .expect("Kernel size should be ok here");

    let layer5 = MaxPool2dLayer::new((2, 2), None, None);

    let shape = layer5.output_dim(layer4.output_dim);
    let dense_0_input_size = shape.0 * shape.1 * shape.2;

    let layer6 = FlattenLayer::new();
    let layer7 = DenseLayer::new(dense_0_input_size, 128, Some(ActivationFunctionType::Relu));
    let layer8 = DenseLayer::new(128, 2, Some(ActivationFunctionType::Sigmoid));

    let mut nn = SequentialModel::new(9);

    nn.push_layer("Conv2D Layer 0".to_string(), Layer::Conv2d(layer0));
    nn.push_layer("MaxPool2D Layer 0".to_string(), Layer::MaxPool2d(layer1));
    nn.push_layer("Conv2D Layer 1".to_string(), Layer::Conv2d(layer2));
    nn.push_layer("MaxPool2D Layer 1".to_string(), Layer::MaxPool2d(layer3));
    nn.push_layer("Conv2D Layer 2".to_string(), Layer::Conv2d(layer4));
    nn.push_layer("MaxPool2D Layer 2".to_string(), Layer::MaxPool2d(layer5));
    nn.push_layer("Flatten Layer".to_string(), Layer::Flatten(layer6));
    nn.push_layer("Dense layer 0".to_string(), Layer::Dense(layer7));
    nn.push_layer(
        "Dense layer 1 (classification)".to_string(),
        Layer::Dense(layer8),
    );

    nn
}

fn main() {
    let input = Array::random((224, 224, 3), Uniform::new(0., 1.));
    let model = create_deep_learning_model(input.shape());
    let start_time = SystemTime::now();
    let result = model.forward(&input);
    let duration = SystemTime::now()
        .duration_since(start_time)
        .unwrap_or_default();
    println!("CNN inference time spent {duration:?}");
    println!("Result: {result:?}");
}
