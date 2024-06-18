use std::io::Read;

use crate::helper::{ImageToMat, MatToSlice, Network};
use anyhow::Result;
use image::DynamicImage;
use itertools::Itertools;
use ncnn_rs::Mat;

pub struct Recognizer {
    net: Network,
    keys: Vec<String>,
}

fn get_key_dict(path: &str) -> Result<Vec<String>> {
    let mut keys = vec!["".to_string()];
    let mut file = std::fs::File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    for line in contents.lines() {
        keys.push(line.to_string());
    }
    keys.push(" ".into());
    Ok(keys)
}


impl Recognizer {
    pub fn init() -> Result<Self> {
        let mut net: Network = Network::load(
            "models/rec/ch_PP-OCRv3_rec.param",
            "models/rec/ch_PP-OCRv3_rec.bin",
            true,
        )?;
        net.set_input_and_output_name("input", "output");

        Ok(Self {
            net,
            keys: get_key_dict("models/rec/ppocr_keys_v1.txt")?,
        })
    }

    pub fn infer(&mut self, img: &DynamicImage) -> Result<(String, f64)> {
        let mat_in = img.to_normalized_mat()?;
        let mat_out = self.net.infer(&mat_in)?;

        // 转四维 vec，维度为：(6625, 8, 1, 1)
        let data = mat_out.to_slice::<f32>().chunks(6625).collect::<Vec<_>>();

        let index_and_prob: Vec<_> = data
            .into_iter()
            .map(|word_prob| {
                let (i, prob) = word_prob
                    .iter()
                    .enumerate()
                    .max_by(|x, y| x.1.total_cmp(y.1))
                    .map(|(i, &prob)| (i, prob))
                    .unwrap_or((0, 1.0));
                (i, prob)
            })
            .collect();

        // 去除*相邻的*重复元素，只保留第一个，并根据索引获取对应的字符
        let res = index_and_prob
            .into_iter()
            .dedup_by(|(i, _), (j, _)| i == j)
            .map(|(i, _)| self.keys[i].clone())
            .join("");

        Ok((res, 1.0))
        }
    
}
