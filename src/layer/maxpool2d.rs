use std::error::Error;

use ndarray::{s, Array, Axis, Ix3};
use ndarray_stats::QuantileExt;

use crate::{activation::ActivationFunctionType, layer::util::add_padding};

pub struct MaxPool2dLayer {
    pub pool_size: (usize, usize),
    pub strides: (usize, usize),
    pub padding: (usize, usize),
}

impl MaxPool2dLayer {
    pub fn new(
        pool_size: (usize, usize),
        strides: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
    ) -> Self {
        let strides = strides.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));
        Self {
            pool_size,
            strides,
            padding,
        }
    }

    pub fn output_dim(&self, input_dim: (usize, usize, usize)) -> (usize, usize, usize) {
        let (input_height, input_width, input_feature_size) = input_dim;
        // TODO: strides are not considered by now, add padding
        (
            input_height / self.pool_size.0,
            input_width / self.pool_size.1,
            input_feature_size,
        )
    }

    pub fn activation_function(&self) -> ActivationFunctionType {
        ActivationFunctionType::None
    }

    pub fn forward(&self, input: &Array<f32, Ix3>) -> Result<Array<f32, Ix3>, Box<dyn Error>> {
        let input_padded = add_padding(input, &self.padding);
        let input_shape = input_padded.shape();
        let input_padded_height = input_shape[0];
        let input_padded_width = input_shape[1];
        let input_padded_feature_size = input_shape[2];

        let (output_height, output_width, output_feature_size) = self.output_dim((
            input_padded_height,
            input_padded_width,
            input_padded_feature_size,
        ));

        let mut output = Array::zeros((output_height, output_width, output_feature_size));

        for (output_row, row) in (0..=input_padded_height - self.pool_size.0)
            .step_by(self.pool_size.0)
            .enumerate()
        {
            let max_row = row + self.pool_size.0;
            for (output_col, col) in (0..=input_padded_width - self.pool_size.1)
                .step_by(self.pool_size.1)
                .enumerate()
            {
                let max_col = col + self.pool_size.1;
                for feature in 0..input_padded_feature_size {
                    let max_feature = feature + 1;
                    let input_slice =
                        input_padded.slice(s![row..max_row, col..max_col, feature..max_feature]);
                    output[[output_row, output_col, feature]] =
                        *input_slice.index_axis(Axis(2), 0).max()?;
                }
            }
        }
        Ok(output)
    }
}
