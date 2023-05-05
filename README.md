# inference_transformer
* 基于transformer的预测，只对于特定模型有效。
* 支持多卡切割模型（多机需要添加多机的信息同步，所以不做了），但是写的多卡比单卡慢（与通讯相关，目前处理的问题比较大）
* 编译依赖于cuda、sentencepiece、nccl，预测依赖于transformer、sentencepiece的模型。
