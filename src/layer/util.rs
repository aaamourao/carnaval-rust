use ndarray::{Array, Ix3, s};

pub fn add_padding(input: &Array<f32, Ix3>, padding: &(usize, usize)) -> Array<f32, Ix3> {
    let input_shape = input.shape();
    let input_channel_size = input_shape[0];
    let input_height = input_shape[1];
    let input_width = input_shape[2];


    let mut input_padded = Array::zeros((input_channel_size,
                                         input_height + padding.0, input_width + padding.1));

    input_padded.slice_mut(s![0..input_channel_size,
        padding.0..input_height, padding.1..input_width]).assign(input);

    return input_padded
}
