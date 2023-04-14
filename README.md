# FastSpeechPPG

from https://github.com/ming024

注：`FastSpeech2/hifigan` 中有2个文件太大放不进来。其中有一个`generator_LJSpeech.pth.tar`是需要的

# tips  
- 训练必须： 

  上传 `generator_LJSpeech.pth.tar` 至 `FastSpeech2/hifigan` 
  
- infer必须：

  上传训练后的模型 `16000.pth.tar` 至 `FastSpeech2/output/ckpt/LJSpeech` 
  
  将命令行参数 --restore_step 设为预训练模型的步数

# Update
2023/3/13 整理了点jupyter notebook  

2023/3/17 
- 移植了tacotron2_ppg的dataloader
  - 与 librosa 0.10.0 有一个地方不兼容，换成0.9.2可解决
- 注释掉了variance adaptor相关内容
- 主要问题在维度问题，目前还在调试。在`config/LJSpeech/model.yaml`中有注明修改内容与原因。

2023/4/5
- 代码全部跑通，可以开始训练。
- 在log里放了个中间数据，总之维度问题解决了。
- 可视化代码还在修改，暂时注释掉不影响训练。

2023/4/7
- 打算用这个sr=16000的hifigan  https://github.com/bshall/hifigan  来配合我们的数据。
- 但是这个hifigan的mel是128channels的。需要修改。
- 新添infer.py，可以load模型进行infer了。

2023/4/8
- 重新训练完了128channels的FastSpeech2模型。（大小为368m）
- infer的结果在 `log/4.8` 里
- 效果还需要改进
  - 提高PPG的准确度
  - 处理FastSpeech的encoder部分
  - 分段padding做streaming infer

2023/4/14
- 正在做streaming infer
