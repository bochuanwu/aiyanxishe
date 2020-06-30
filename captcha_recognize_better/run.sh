python3 train.py --train_data result/train/ --valid_data result/test/  --rgb --Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn --sensitive
#python3 train.py --train_data result/train/ --valid_data result/test/ --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn  --adam --lr 0.001
