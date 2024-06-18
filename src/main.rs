mod det;
mod rec;
mod rect;

#[macro_use]
mod helper;
use anyhow::Result;
use helper::*;
use image::{DynamicImage, GenericImageView};
use rec::Recognizer;

fn resize_image(image: &DynamicImage) -> DynamicImage {
    let (width, height) = image.dimensions();
    // 把 height 调整为 48，宽度按原先比例缩放
    let factor = width as f32 / height as f32;
    let height = 48;
    let width = (height as f32 * factor) as u32;
    image.resize_exact(width, height, image::imageops::FilterType::Lanczos3)
}

fn main() {
    test("inputs/c.png").unwrap();
}

fn test(path:&str) -> Result<()> {
    // 读取照片
    let mut det = det::Detector::init()?;
    let img = image::open(path)?;

    let (boxes, imgs, scores) = det.infer(&img)?;

    let mut rec = Recognizer::init()?;
    for (i, (img, score)) in imgs.iter().zip(scores).enumerate() {
        img.save(format!("outputs/c-det-clip/{i}-{score:.2}.png"))?;
        let (res, prob) = rec.infer(&resize_image(img))?;
        println!("{i}: {}", res);
    }

    Ok(())
}
