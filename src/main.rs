mod det;
mod helper;
mod rect;
use anyhow::Result;
use helper::*;

fn main() -> Result<()> {
    // 读取照片
    let mut det = det::Detector::init()?;
    let img = image::open("inputs/a.png")?;

    let res = det.infer(img)?.to_gray_img()?;
    res.save("outputs/a-det.png")?;

    Ok(())
}
