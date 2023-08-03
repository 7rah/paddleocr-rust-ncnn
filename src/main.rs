mod det;

#[macro_use]
mod helper;
mod rect;
use anyhow::Result;
use helper::*;

fn main() -> Result<()> {
    // 读取照片
    let mut det = det::Detector::init()?;
    let img = image::open("inputs/a.png")?;

    let (boxes, imgs, scores) = det.infer(img)?;

    for (i, (img, score)) in imgs.iter().zip(scores).enumerate() {
        img.save(format!("outputs/a-det-clip/{i}-{score:.2}.png"))?;
    }

    Ok(())
}
