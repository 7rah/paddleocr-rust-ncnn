use imageproc::point::Point;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Rotation {
    sin_theta: f64,
    cos_theta: f64,
}

impl Rotation {
    /// A rotation of `theta` radians.
    pub fn new(theta: f64) -> Rotation {
        let (sin_theta, cos_theta) = theta.sin_cos();
        Rotation {
            sin_theta,
            cos_theta,
        }
    }
}

pub trait Rotate {
    fn rotate(&self, rotation: Rotation) -> Point<f64>;
    fn invert_rotation(&self, rotation: Rotation) -> Point<f64>;
}

impl Rotate for Point<f64> {
    /// Rotates a point.
    fn rotate(&self, rotation: Rotation) -> Point<f64> {
        let x = self.x * rotation.cos_theta + self.y * rotation.sin_theta;
        let y = self.y * rotation.cos_theta - self.x * rotation.sin_theta;
        Point::new(x, y)
    }

    /// Inverts a rotation.
    fn invert_rotation(&self, rotation: Rotation) -> Point<f64> {
        let x = self.x * rotation.cos_theta - self.y * rotation.sin_theta;
        let y = self.y * rotation.cos_theta + self.x * rotation.sin_theta;
        Point::new(x, y)
    }
}
