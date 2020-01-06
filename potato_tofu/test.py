from PIL import Image
import os
import torch
import pandas as pd
from torchvision import datasets, models, transforms
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
transform = transforms.Compose([
        transforms.Scale(299),
        #transforms.RandomResizedCrop(224),
        transforms.CenterCrop(224),
        #transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def get_files(directory):
    return [os.path.join(directory, f) for f in list(os.listdir(directory))]


files = get_files('./test')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('./model.pkl')  # 直接加载模型
model = model.to(device)
results=[]
ids = []
for i, item in enumerate(files):
    print('Processing %i of %i (%s)' % (i+1, len(files), item))
    image = transform(Image.open(item).convert("RGB"))
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        py = model(image)
        _, predicted = torch.max(py, 1)  # 获取分类结果
        classIndex_ = predicted.cpu().numpy()[0]
        print('预测结果',item.split('/')[-1][0:-4], classIndex_)
        results.append(classIndex_)
        ids.append(int(item.split('/')[-1][0:-4]))
results = pd.Series(results)
submission = pd.concat([pd.Series(ids),results],axis=1)
submission = submission.sort_values(axis=0,by=0)
submission.to_csv('./submission.csv',index=False)





