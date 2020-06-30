CUDA_VISIBLE_DEVICES=1 python3 demo.py \
--Transformation TPS --rgb --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder demo_image/ --sensitive \
--saved_model saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth
