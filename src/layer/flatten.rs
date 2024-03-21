use ndarray::{Array, array, Axis, IntoDimension, Ix3};
use crate::activation::ActivationFunctionType;
use crate::layer::{Layer, LayerError, LayerType};

pub struct Flatten {
    pub has_channels: bool,
}

impl Flatten {
    pub fn new (has_channels: Option<bool>) -> Self {
        let has_channels = has_channels.unwrap_or(false);
        Flatten {
           has_channels
        }
    }

    fn forward_channels(&self, input: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>, LayerError> {
        todo!()
        /*
        let input_shape = input.shape();
        let channels = input_shape[0];
        let input_height = input_shape[1];
        let input_width = input_shape[2];

        let mut result = Array::zeros((channels, 1, input_height * input_width));

        for channel in 0..channels {
            let mut flatten_input = Array::from_iter(input.iter().cloned());
            let flatten_input_size = flatten_input.shape()[0];
            result[channel].index_axis_mut(Axis(0), 0) = flatten_input.view_mut();
        }

        return Ok(result)
        */
    }

    fn forward_with_no_channels(&self, input: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>, LayerError> {
        let mut flatten_input = Array::from_iter(input.iter().cloned());
        let flatten_input_size = flatten_input.shape()[0];
        Ok(flatten_input.into_shape((1, flatten_input_size, 1)).unwrap().to_owned())
    }
}

impl Layer for Flatten {
    fn get_layer_type(&self) -> LayerType {
        LayerType::Flatten
    }

    fn get_activation_function(&self) -> ActivationFunctionType {
        ActivationFunctionType::None
    }

    fn forward(&self, input: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>, LayerError> {
        if self.has_channels {
            return self.forward_channels(input)
        }
        return self.forward_with_no_channels(input)
    }
}