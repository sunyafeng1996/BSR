import registry,datafree
import torch
import torch.utils.data as Data
import numpy as np
import math
from PIL import Image
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


root2s=['bsr-r34r18c10-omz8']

root1='/wxw2/syf/projects/BSR/run/'

model_name='resnet34'  # model
dataset='cifar10'  # dataset
data_root='datasets/' #dataset path
batch_size=256

# load
num_classes, ori_dataset, val_dataset = registry.get_dataset(name=dataset, data_root=data_root)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size, shuffle=False,num_workers=0, pin_memory=True)
evaluator = datafree.evaluators.classification_evaluator(val_loader)

normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[dataset])
model = registry.get_model(model_name, num_classes=num_classes, pretrained=True).eval()
model.load_state_dict(torch.load('/wxw2/syf/projects/BSR/checkpoints/pretrained/%s_%s.pth'%(dataset, model_name), map_location='cpu')['state_dict'])
model = model.cuda()
model.eval()

for root2 in root2s:
    print('Start process {}'.format(root2))
    root=root1+root2+'/dataset'
    data_train = ImageFolder(root, transform=ori_dataset.transform)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True,num_workers=4)
    max_samples=len(train_loader.dataset)
    now_sample=0
    for i,(images,targets) in enumerate(train_loader):
            images=images.cuda()
            # targets=targets.cuda()
            output,feature=model(images,return_features=True)
            if i==0:
                all_features=feature.detach().cpu().numpy()
                all_targets=targets.detach().cpu().numpy()
            else:
                all_features=np.vstack((all_features, feature.detach().cpu().numpy())) 
                all_targets=np.hstack((all_targets, targets.detach().cpu().numpy())) 
            
            now_sample=now_sample+images.size(0)
            print("[Processed samples: {}]/[Max samples: {}]".format(now_sample,max_samples), end='\r')
    
    print('')     
    # tsne features
    from sklearn.manifold import TSNE
    tsne = TSNE(2)
    print('Start TSNE')
    tsne_features = tsne.fit_transform(all_features)
    del tsne
    
    pd.DataFrame(tsne_features).to_csv('/wxw2/syf/projects/BSR/visualization/points/'+root2+'_points.csv')
    pd.DataFrame(all_targets).to_csv('/wxw2/syf/projects/BSR/visualization/points/'+root2+'_targets.csv')
    print('The points is saved in {}'.format('/wxw2/syf/projects/BSR/visualization/points/'+root2+'_points.csv'))