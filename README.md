# FastSpeechPPG

from https://github.com/ming024

注：`FastSpeech2/hifigan` 中有2个文件太大放不进来。其中有一个`generator_LJSpeech.pth.tar`是需要的

# Update
2023/3/13 整理了点jupyter notebook  

2023/3/17 
- 移植了tacotron2_ppg的dataloader
  - 与 librosa 0.10.0 有一个地方不兼容，换成0.9.2可解决
- 注释掉了variance adaptor相关内容
- 主要问题在维度问题，目前还在调试。在`config/LJSpeech/model.yaml`中有注明修改内容与原因。

          
