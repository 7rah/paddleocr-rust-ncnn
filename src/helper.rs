use anyhow::{Context, Result};
use image::{self, DynamicImage, GenericImageView, ImageBuffer, Luma};
use ncnn_rs::{Mat, Net};

pub trait ImageToMat {
    fn to_normalized_mat(&self) -> Result<Mat>;
}

pub trait MatToGrayImg {
    fn to_gray_img(&self) -> Result<DynamicImage>;
}

pub trait MatToSlice {
    fn to_slice<T>(&self) -> &[T];
    fn to_2d_slice<T>(&self) -> &[&[T]];
}

impl MatToSlice for Mat {
    fn to_slice<T>(&self) -> &[T] {
        unsafe {
            core::slice::from_raw_parts(self.data() as *const T, (self.w() * self.h()) as usize)
        }
    }

    fn to_2d_slice<T>(&self) -> &[&[T]] {
        self.to_slice()
    }
}

impl MatToGrayImg for Mat {
    fn to_gray_img(&self) -> Result<DynamicImage> {
        let data: &[f32] = self.to_slice();

        let img: ImageBuffer<Luma<u8>, Vec<u8>> = image::ImageBuffer::from_vec(
            self.w() as u32,
            self.h() as u32,
            data.into_iter()
                .map(|x| (x * 255.0) as u8)
                .into_iter()
                .collect(),
        )
        .with_context(|| "buffer is not big enough")?;

        let img = DynamicImage::ImageLuma8(img);

        Ok(img)
    }
}

impl ImageToMat for DynamicImage {
    fn to_normalized_mat(&self) -> Result<Mat> {
        let (width, height) = self.dimensions();
        let img = self.to_rgb8();

        let mut mat = Mat::from_pixels(
            img.as_raw(),
            ncnn_rs::MatPixelType::RGB,
            width as i32,
            height as i32,
            None,
        )?;

        // normalize
        let mean = [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0];
        let norm = [
            1.0 / 0.229 / 255.0,
            1.0 / 0.224 / 255.0,
            1.0 / 0.225 / 255.0,
        ];
        mat.substract_mean_normalize(&mean, &norm);
        Ok(mat)
    }
}

pub struct Network {
    net: ncnn_rs::Net,
    input: String,
    output: String,
}

impl Network {
    pub fn load(param: &str, bin: &str, enable_vulkan: bool) -> Result<Self> {
        let mut net = Net::new();

        let mut opt = ncnn_rs::Option::new();
        opt.set_vulkan_compute(enable_vulkan);
        net.set_option(&opt);

        net.load_param(param)?;
        net.load_model(bin)?;

        Ok(Self {
            net,
            input: "".to_string(),
            output: "".to_string(),
        })
    }
    pub fn load_from_memory(param: &[u8], bin: &[u8], enable_vulkan: bool) -> Result<Self> {
        let mut net = Net::new();

        let mut opt = ncnn_rs::Option::new();
        opt.set_vulkan_compute(enable_vulkan);
        net.set_option(&opt);

        //net.load_param_memory(param)?;
        //net.load_model_memory(bin)?;

        Ok(Self {
            net,
            input: "".to_string(),
            output: "".to_string(),
        })
    }

    pub fn set_input_and_output_name(&mut self, input: &str, output: &str) {
        self.input = input.to_string();
        self.output = output.to_string();
    }

    pub fn infer(&mut self, mat: &Mat) -> Result<Mat> {
        let mut extractor = self.net.create_extractor();
        let mut mat_out = Mat::new();
        extractor.input(&self.input, mat)?;
        extractor.extract(&self.output, &mut mat_out)?;
        Ok(mat_out)
    }
}
