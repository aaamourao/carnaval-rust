use ndarray::{Array, Axis, Ix3, s};
use ndarray_stats::QuantileExt;
use crate::activation::ActivationFunctionType;
use crate::layer::{Layer, LayerError, LayerType};
use crate::layer::util::add_padding;

pub struct MaxPool2D {
    pub pool_size: (usize, usize),
    pub strides: (usize, usize),
    pub padding: (usize, usize),
}

impl MaxPool2D {
    pub fn new(pool_size: (usize, usize), strides: Option<(usize, usize)>,
               padding: Option<(usize, usize)>) -> Self {
        let strides = strides.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));
        MaxPool2D {
            pool_size,
            strides,
            padding,
        }
    }

    pub fn get_output_dim(&self, input_dim: (usize, usize, usize))-> (usize, usize, usize) {
        let (input_depth, input_height, input_width) = input_dim;
        // TODO: strides are not considered by now
        (input_depth, input_height - self.pool_size.0 + 1, input_width - self.pool_size.1 + 1)
    }
}

impl Layer for MaxPool2D {
    fn get_layer_type(&self) -> LayerType {
        LayerType::MaxPool
    }

    fn get_activation_function(&self) -> ActivationFunctionType {
        ActivationFunctionType::None
    }

    fn forward(&self, input: &Array<f64, Ix3>) -> Result<Array<f64, Ix3>, LayerError> {
        let input_padded = add_padding(input, &self.padding);
        let input_shape = input_padded.shape();
        let input_padded_depth = input_shape[0];
        let input_padded_height = input_shape[1];
        let input_padded_width = input_shape[2];

        let (output_depth, output_height, output_width) = self.get_output_dim(
            (input_padded_depth, input_padded_height, input_padded_width));

        let mut output = Array::zeros((output_depth, output_height, output_width));

        for channel in 0..input_padded_depth {
            let max_depth = channel + 1;
            for row in 0..input_padded_height - self.pool_size.0 + 1 {
                // TODO: for now, dilatation/stride is not being considered in the convolution
                let max_row = row + self.pool_size.0;
                for col in 0..input_padded_width - self.pool_size.1 + 1{
                    let max_col = col + self.pool_size.1;
                    let input_slice = input_padded.slice(
                        s![channel..max_depth, row..max_row, col..max_col]);
                    output[[channel, row, col]] = *input_slice.index_axis(Axis(0), 0).max().unwrap();
                }
            }
        }

        Ok(output)
    }
}
