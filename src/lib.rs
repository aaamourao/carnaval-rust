mod activation;
mod layer;

use activation::{relu, sigm};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_works() {
        let result = relu(-2.0);
        let result_bigger_than_0 = relu(3.0);
        assert_eq!(result, 0.0);
        assert_eq!(result_bigger_than_0, 3.0);
    }

    #[test]
    fn sigm_works() {
        let result = sigm(0.0);
        let result_6 = sigm(6.0);
        let result_minus_6 = sigm(-6.0);
        assert_eq!(result, 0.5);
        assert_eq!(result_6, 0.9975273768433653);
        assert_eq!(result_minus_6, 0.0024726231566347743);
    }
}
