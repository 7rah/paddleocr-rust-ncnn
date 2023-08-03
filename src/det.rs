use std::{cmp::Ordering, default};

use crate::helper::*;
use crate::rect::{min_area_rect, Rotate, Rotation};
use anyhow::{Context, Result};
use derivative::Derivative;
use image::{imageops, DynamicImage, GrayImage, ImageBuffer, Luma, Pixel, Rgb, RgbImage, Rgba};
use imageproc::drawing::Canvas;
use imageproc::{
    contours::{find_contours, Contour},
    definitions::HasWhite,
    drawing::draw_polygon_mut,
    point::Point,
};
use ncnn_rs::Mat;
use num_traits::Bounded;
use num_traits::{real::Real, Num, NumCast, Pow};
use rust_embed::RustEmbed;
use visioncortex::{PerspectiveTransform, Point2};

#[derive(RustEmbed)]
#[folder = "models/det/"]
struct Assert;

#[derive(Debug, Derivative)]
#[derivative(Default)]
struct DetectorConfig {
    #[derivative(Default(value = "0.3"))]
    box_threshold: f64,
    #[derivative(Default(value = "1.6"))]
    unclip_ratio: f64,
    #[derivative(Default(value = "4.0"))]
    max_degree: f64, // 角度制
}

// 偏差角度翻译

pub struct Detector {
    net: Network,
}

// 写一个宏统计时间，打印当前行号
macro_rules! timeit {
    ($e:expr) => {{
        let start = std::time::Instant::now();
        let res = $e;
        let end = std::time::Instant::now();
        println!(
            "timeit: {}:{}:{}:  {:?}",
            file!(),
            line!(),
            column!(),
            end.duration_since(start)
        );
        res
    }};
}

impl Detector {
    pub fn init() -> Result<Self> {
        //let param = Assert::get("ch_PP-OCRv3_det_infer.param").with_context(|| "Failed to load param")?;
        //let bin = Assert::get("ch_PP-OCRv3_det_infer.bin").with_context(|| "Failed to load bin")?;
        // let mut net = Network::load_from_memory(&param.data, &bin.data, true)?;
        let mut net: Network = Network::load(
            "models/det/ch_PP-OCRv3_det_infer.param",
            "models/det/ch_PP-OCRv3_det_infer.bin",
            true,
        )?;
        net.set_input_and_output_name("input", "output");

        Ok(Self { net })
    }

    pub fn infer(&mut self, img: DynamicImage) -> Result<Mat> {
        let config = DetectorConfig::default();

        // infer
        let mat_in = img.to_normalized_mat()?;
        let mat_out = self.net.infer(&mat_in)?;

        let (boxes, _scores) = get_boxes_and_scores(&mat_out, config.box_threshold)?;
        // boxes 转 f64

        for (i, rect) in boxes.iter().enumerate() {
            // expand box
            let distance = get_contour_area(rect, config.unclip_ratio);
            let rect = expand_box(&rect, distance);

            // clip image
            let clipped_img = clip(&img, &rect, config.max_degree);
            clipped_img.save(format!("outputs/a-det-clip/{i}.png"))?;
        }

        Ok(mat_out)
    }
}

fn get_contour_area(box_points: &[Point<f64>], unclip_ratio: f64) -> f64 {
    let pts_num = 4;
    let mut area = 0.0;
    let mut dist = 0.0;

    for i in 0..pts_num {
        let next_i = (i + 1) % pts_num;

        area += box_points[i].x * box_points[next_i].y - box_points[i].y * box_points[next_i].x;

        let dx = box_points[i].x - box_points[next_i].x;
        let dy = box_points[i].y - box_points[next_i].y;
        dist += f64::sqrt(dx * dx + dy * dy);
    }

    area = f64::abs(area / 2.0);

    let distance = area * unclip_ratio / dist;
    distance
}

fn distance<T>(p1: &Point<T>, p2: &Point<T>) -> T
where
    T: Num + Real,
{
    let x = p1.x - p2.x;
    let y = p1.y - p2.y;
    (x * x + y * y).sqrt()
}

fn clip(img: &DynamicImage, rect: &[Point<f64>; 4], max_degree: f64) -> DynamicImage {
    let dx = rect[1].x - rect[0].x;
    let dy = rect[1].y - rect[0].y;
    let degree = dy.atan2(dx).to_degrees();

    if degree.abs() < max_degree {
        clip_simple(img, rect)
    } else {
        println!("adjust angle clip");
        clip_adjust_angle(img, rect)
    }
}

fn clip_simple(img: &DynamicImage, rect: &[Point<f64>; 4]) -> DynamicImage {
    let (min_x, max_x, min_y, max_y) = get_min_max_x_y(rect);

    let cropped_img = imageops::crop_imm(
        img,
        min_x.round() as u32,
        min_y.round() as u32,
        (max_x - min_x).round() as u32,
        (max_y - min_y).round() as u32,
    );
    DynamicImage::ImageRgba8(cropped_img.to_image())
}

fn clip_adjust_angle(img: &DynamicImage, rect: &[Point<f64>; 4]) -> DynamicImage {
    // 图中的四个点，按照左上、右上、右下，左下的顺序
    let src_points = rect.iter().map(|p| [p.x, p.y]).flatten().collect();

    // 部分矩形与水平线的夹角不为 0，需要我们进行旋转修正，先计算夹角
    let dx = rect[1].x - rect[0].x;
    let dy = rect[1].y - rect[0].y;
    let angle = dy.atan2(dx);

    // 计算旋转后的四个点
    let rotation = Rotation::new(angle);
    let rotated_rect: Vec<_> = rect.iter().map(|p| p.rotate(rotation)).collect();

    let (min_x, max_x, min_y, max_y) = get_min_max_x_y(&rotated_rect);
    let offset_rect: Vec<_> = rotated_rect
        .iter()
        .map(|p| Point {
            x: p.x - min_x,
            y: p.y - min_y,
        })
        .collect();

    println!("offset_rect: {:?}", offset_rect);

    let dst_points = offset_rect.iter().map(|p| [p.x, p.y]).flatten().collect();

    // 计算透视变换矩阵
    let transform = PerspectiveTransform::new(src_points, dst_points);

    // 创建目标图像
    let mut dst_img = RgbImage::new((max_x - min_x + 1.0) as u32, (max_y - min_y + 1.0) as u32);

    for (x, y, pixel) in dst_img.enumerate_pixels_mut() {
        // 计算原图像中的坐标
        let Point2 { x: src_x, y: src_y } =
            transform.transform_inverse(visioncortex::Point2::new(x as f64, y as f64));

        let src_x = src_x.round() as u32;
        let src_y = src_y.round() as u32;

        //// 获取原图像中的像素
        let src_pixel = img.get_pixel(src_x, src_y).to_rgb();
        *pixel = src_pixel;
    }

    DynamicImage::from(dst_img)
}

// 计算边上像素点的平均值
fn calc_edge_avg_pixel(img: &DynamicImage, boxes: &[Point<f64>; 4]) -> Rgb<u8> {
    // 计算长宽
    let width: f64 = distance(&boxes[0], &boxes[1]);
    let height: f64 = distance(&boxes[0], &boxes[3]);

    let src_points = boxes.iter().map(|p| [p.x, p.y]).flatten().collect();
    let dst_points = vec![0.0, 0.0, width, 0.0, width, height, 0.0, height];

    // 计算透视变换矩阵
    let transform = PerspectiveTransform::new(src_points, dst_points);
    let (width, height) = (width as u32, height as u32);

    // 计算四条边上的像素点的坐标
    let top = (0..width + 1).map(|x| (x, 0));
    let bottom = (0..width + 1).map(|x| (x, height));
    let left = (1..height).map(|y| (0, y));
    let right = (1..height).map(|y| (width, y));

    // 把四个边上的点合并
    let points = top.chain(bottom).chain(left).chain(right).map(|(x, y)| {
        let p = transform.transform_inverse(Point2::new(x as f64, y as f64));
        let pixel = img.get_pixel(p.x as u32, p.y as u32).to_rgb();
        pixel
    });

    let cnt = (width + height) * 2;

    // 计算平均值
    let sum = points.fold((0, 0, 0), |acc, pixel| {
        (
            acc.0 + pixel[0] as u32,
            acc.1 + pixel[1] as u32,
            acc.2 + pixel[2] as u32,
        )
    });

    Rgb([
        (sum.0 / cnt) as u8,
        (sum.1 / cnt) as u8,
        (sum.2 / cnt) as u8,
    ])
}

// rust 计算三个 point 形成的角的角度
fn calc_angle(p1: &Point<f64>, p2: &Point<f64>, p3: &Point<f64>) -> f64 {
    let v1 = (p1.x - p2.x, p1.y - p2.y);
    let v2 = (p3.x - p2.x, p3.y - p2.y);

    let dot = v1.0 * v2.0 + v1.1 * v2.1;
    let det = v1.0 * v2.1 - v1.1 * v2.0;

    let angle = det.atan2(dot);

    // 变成角度
    angle.to_degrees()
}

fn expand_box<T>(rectangle: &[Point<T>; 4], distance: T) -> [Point<T>; 4]
where
    T: Num + Real,
{
    let calc_expand_vector = |target: &Point<T>, near: &Point<T>| {
        let dx = target.x - near.x;
        let dy = target.y - near.y;
        let length = (dx * dx + dy * dy).sqrt();

        (dx / length * distance, dy / length * distance)
    };

    let do_expand = |target, near1, near2| {
        let unit_vector1 = calc_expand_vector(target, near1);
        let unit_vector2 = calc_expand_vector(target, near2);

        // target + unit_vector1 + unit_vector2
        Point {
            x: (target.x + unit_vector1.0 + unit_vector2.0),
            y: (target.y + unit_vector1.1 + unit_vector2.1),
        }
    };

    [
        do_expand(&rectangle[0], &rectangle[3], &rectangle[1]),
        do_expand(&rectangle[1], &rectangle[0], &rectangle[2]),
        do_expand(&rectangle[2], &rectangle[1], &rectangle[3]),
        do_expand(&rectangle[3], &rectangle[2], &rectangle[0]),
    ]
}

fn get_boxes_and_scores(mat: &Mat, threshold: f64) -> Result<(Vec<[Point<f64>; 4]>, Vec<f64>)> {
    let img = mat.to_gray_img()?.into_luma8();

    let (rects, scores): (Vec<_>, Vec<_>) = find_contours::<u32>(&img)
        .into_iter()
        .filter_map(|Contour { points, .. }| {
            if points.len() < 4 {
                return None;
            }

            let rect = min_area_rect(&points);
            let score = calc_score_slow(&mat, &points);

            if score < threshold {
                return None;
            }

            Some((rect, score))
        })
        .unzip();

    Ok((rects, scores))
}

fn calc_score_slow(mat: &Mat, point: &[Point<u32>]) -> f64 {
    let (min_x, max_x, min_y, max_y) = get_min_max_x_y(&point);

    let bitmap: &[f32] = mat.to_slice();
    let mut mask = GrayImage::new(max_x - min_x + 1, max_y - min_y + 1);
    let moved_point: Vec<_> = point
        .iter()
        .map(|p| Point::new((p.x - min_x) as i32, (p.y - min_y) as i32))
        .collect();
    draw_polygon_mut(&mut mask, &moved_point, Luma([1u8]));

    let iter = mask.enumerate_pixels().map(|(x, y, p)| {
        if p.0[0] == 1 {
            bitmap[((x + min_x) + (y + min_y) * mat.w() as u32) as usize]
        } else {
            0.0
        }
    });

    let cnt = iter.len();
    let sum: f32 = iter.sum();
    let mean = sum as f64 / cnt as f64;

    mean
}

fn get_min_max_x_y<T>(point: &[Point<T>]) -> (T, T, T, T)
where
    T: Num + PartialOrd + Copy + Bounded,
{
    fn max<T: PartialOrd>(a: T, b: T) -> T {
        if b > a {
            b
        } else {
            a
        }
    }

    fn min<T: PartialOrd>(a: T, b: T) -> T {
        if b > a {
            a
        } else {
            b
        }
    }

    point.iter().fold(
        (
            T::max_value(),
            T::min_value(),
            T::max_value(),
            T::min_value(),
        ),
        |acc, p| {
            (
                min(acc.0, p.x),
                max(acc.1, p.x),
                min(acc.2, p.y),
                max(acc.3, p.y),
            )
        },
    )
}
