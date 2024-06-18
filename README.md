# 背景
paddleocr 是目前开源的，中文识别效果较好，而且模型推理速度也比较快的 ocr 模型。但是 paddleocr 官方只有 python 版本，需要安装大量的依赖，
其 CPU 推理速度也较慢，而现有的 RapidOCR 项目是基于 openvino 和 onnxruntime 重新实现的，这两者都不好使用 GPU 推理，onnxruntime 使用 
CPU 推理大约耗时 20s。而 ncnn 是一个轻量，高性能的神经网络推理框架，而且支持基于 vulkan compute shader 的 GPU 加速，
这使得我们能以较低的门槛使用 GPU 加速 paddleocr。本项目使用 rust-ncnn 作为推理框架，参考了 RapidOCR 项目中的推理代码，使用 Rust 重新实现了
推理流程，项目无 opencv 依赖，除了 ncnn 外，其余的依赖都能比较容易地跨平台。

# 编译运行
参考 [rust-ncnn](https://github.com/tpoisonooo/rust-ncnn) 配置环境，环境配置好后（能编译 rust-ncnn），直接运行下面代码即可

```
cargo run --release
```

# 原理
paddleocr 项目中有三个模型，相关的效果可以去看 outputs 目录下的图片。
* 第一个模型负责识别当前图片中的文字区域，输出为灰度图。我们会将这些有文字的区域裁切出来，送入下面的模型处理。
* 第二个模型负责识别文字条是否颠倒了（如颠倒 90 180 度），然后将其旋转过来。
* 第三个模型负责识别文字条中的文字，输出为 (6625, 8, 1, 1) 的四维矩阵。我们通过查表 ppocr_keys_v1.txt 文件里面相关 index 对应的文字，即可将矩阵转换为文字。具体的实现可以去看 src/rec.rs 下面
