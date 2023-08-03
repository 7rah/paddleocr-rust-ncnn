use imageproc::{geometry::convex_hull, point::Point};
use num_traits::NumCast;

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

pub fn min_area_rect(points: &[Point<u32>]) -> [Point<f64>; 4] {
    let hull: Vec<_> = convex_hull(points)
        .iter()
        .map(|p| Point::new(p.x as f64, p.y as f64))
        .collect();
    match hull.len() {
        0 => panic!("no points are defined"),
        1 => [hull[0]; 4],
        2 => [hull[0], hull[1], hull[1], hull[0]],
        _ => rotating_calipers(&hull),
    }
}

fn rotating_calipers(points: &[Point<f64>]) -> [Point<f64>; 4] {
    let mut edge_angles: Vec<f64> = points
        .windows(2)
        .map(|e| {
            let edge = e[1] - e[0];
            ((edge.y.atan2(edge.x) + std::f64::consts::PI) % (std::f64::consts::PI / 2.)).abs()
        })
        .collect();

    edge_angles.dedup();

    let mut min_area = f64::MAX;
    let mut res = vec![Point::new(0.0, 0.0); 4];
    for angle in edge_angles {
        let rotation = Rotation::new(angle);
        let rotated_points: Vec<Point<f64>> = points.iter().map(|p| p.rotate(rotation)).collect();

        let (min_x, max_x, min_y, max_y) =
            rotated_points
                .iter()
                .fold((f64::MAX, f64::MIN, f64::MAX, f64::MIN), |acc, p| {
                    (
                        acc.0.min(p.x),
                        acc.1.max(p.x),
                        acc.2.min(p.y),
                        acc.3.max(p.y),
                    )
                });

        let area = (max_x - min_x) * (max_y - min_y);
        if area < min_area {
            min_area = area;
            res[0] = Point::new(max_x, min_y).invert_rotation(rotation);
            res[1] = Point::new(min_x, min_y).invert_rotation(rotation);
            res[2] = Point::new(min_x, max_y).invert_rotation(rotation);
            res[3] = Point::new(max_x, max_y).invert_rotation(rotation);
        }
    }

    res.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());

    let i1 = if res[1].y > res[0].y { 0 } else { 1 };
    let i2 = if res[3].y > res[2].y { 2 } else { 3 };
    let i3 = if res[3].y > res[2].y { 3 } else { 2 };
    let i4 = if res[1].y > res[0].y { 1 } else { 0 };

    [
        Point::new(res[i1].x.floor(), res[i1].y.floor()),
        Point::new(res[i2].x.ceil(), res[i2].y.floor()),
        Point::new(res[i3].x.ceil(), res[i3].y.ceil()),
        Point::new(res[i4].x.floor(), res[i4].y.ceil()),
    ]
}
