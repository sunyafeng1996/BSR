import torch
import numpy as np  
from sklearn.neighbors import LocalOutlierFactor  
import os
import math

def get_num_label(data_pool,num_classes):
    num_labels=torch.zeros((num_classes))
    for i in range(num_classes):
        if data_pool.features_pool[i] !=None:
            num_labels[i]=len(data_pool.features_pool[i])
        else:
            num_labels[i]=0
    return num_labels

def balanced_label_generator(num_labels,batch_size):
    diff=num_labels.max()-num_labels
    # diff=1/num_labels
    normalized_probabilities = diff / diff.sum()
    cumulative_probabilities = normalized_probabilities.cumsum(dim=0)   
    random_numbers = torch.rand(batch_size)   
    selected_labels = torch.searchsorted(cumulative_probabilities, random_numbers, right=True)
    
    return selected_labels

def balanced_within_category(data_pool,num_classes,batch_size): 
    max_del=math.ceil(batch_size/num_classes)
  
    for i in range(num_classes):
        
        if data_pool.features_pool[i] !=None:
            if len(data_pool.features_pool[i])>10:
                features=None
                for k,v in data_pool.features_pool[i].items():
                    if features is None:
                        features=v
                        keys=[k]
                    else:
                        features=np.vstack((features,v))
                        keys.append(k)


                lof = LocalOutlierFactor(n_neighbors=10,contamination=0.01)
                y_pred = lof.fit_predict(features)
                outlier_scores = -lof.negative_outlier_factor_
                

                q1 = np.percentile(outlier_scores, 25)  
                q3 = np.percentile(outlier_scores, 75)

                iqr = q3 - q1  

                lower_bound = q1 - 1.5 * iqr  
                upper_bound = q3 + 1.5 * iqr

                outliers_index = np.where((outlier_scores > upper_bound) | (outlier_scores < lower_bound) )[0]
                
                c_del=0
                for del_index in outliers_index:
                    if c_del<=max_del:
                        key=keys[del_index]
                        os.remove(key)
                        data_pool.features_pool[i].pop(key)
                        c_del=c_del+1
                    else:
                        break
                del lof
        
    return data_pool