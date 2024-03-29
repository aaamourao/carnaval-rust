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
        let (input_height, input_width, input_feature_size) = input_dim;
        // TODO: strides are not considered by now, add padding
        (input_height / self.pool_size.0, input_width / self.pool_size.1, input_feature_size)
    }
}

impl Layer for MaxPool2D {
    fn get_layer_type(&self) -> LayerType {
        LayerType::MaxPool
    }

    fn get_activation_function(&self) -> ActivationFunctionType {
        ActivationFunctionType::None
    }

    fn forward(&self, input: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>, LayerError> {
        let input_padded = add_padding(input, &self.padding);
        let input_shape = input_padded.shape();
        let input_padded_height = input_shape[0];
        let input_padded_width = input_shape[1];
        let input_padded_feature_size = input_shape[2];

        let (output_height, output_width, output_feature_size) = self.get_output_dim(
            (input_padded_height, input_padded_width, input_padded_feature_size));

        let mut output = Array::zeros((output_height, output_width, output_feature_size));

        let mut output_row = 0;
        for row in (0..(input_padded_height - self.pool_size.0 + 1)).step_by(self.pool_size.0) {
            let max_row = row + self.pool_size.0;
            let mut output_col = 0;
            for col in
            (0..(input_padded_width - self.pool_size.1 + 1)).step_by(self.pool_size.1) {
                let max_col = col + self.pool_size.1;
                for feature in 0..input_padded_feature_size {
                    let max_feature = feature + 1;
                    let input_slice = input_padded.slice(
                        s![row..max_row, col..max_col, feature..max_feature]);
                    output[[output_row, output_col, feature]] =
                        *input_slice.index_axis(Axis(2), 0).max().unwrap();
                }
                output_col += 1;
            }
            output_row += 1;
        }

        Ok(output)
    }
}
