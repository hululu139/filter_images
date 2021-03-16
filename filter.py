# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:00:49 2021

@author: Asus
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from datetime import timedelta
from datetime import datetime
import os
import shutil
root_dir="pipeline_output"

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err
def compare_images(imageA, imageB):
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB, multichannel=True)
    return m,s
#extract the file name and picture to a new folder and csv
user_df=pd.read_csv('pipeline_output/user_data.csv')
image_df=pd.read_csv('pipeline_output/image_data.csv')
all_df=image_df.merge(user_df, left_on='image_id', right_on='image_id')
all_df['inform']=all_df['user_ID'].astype(str) + all_df["location"]
userinform=list((all_df.inform.unique()))
user=list((all_df.user_ID.unique()))
ms1=list()
for u in user:
    os.makedirs(root_dir+'/filtered_data_for_{}'.format(u)) #if you want to store the images in different folder
    store_dir=root_dir+'/filtered_data_for_{}'.format(u)
    filter_df=pd.DataFrame(columns=['raw_image_filename','curret_bbox','current_annotation_file','current_person_count','curret_posture'])
    for i in userinform:
        dfa=all_df[all_df.inform==i]
        dfa['time_stamp']=dfa['time_stamp'].astype('str')
        dfa['time_stamp1']=dfa.time_stamp.apply(lambda x:datetime.strptime(x, '%m-%d-%Y,%H:%M:%S+%f'))
        difference=pd.DataFrame(dfa.time_stamp1.diff())
        difference.columns=['time_stamp_diff']
        diff_up=difference[difference['time_stamp_diff']>timedelta(minutes=45)]
        time_leak=diff_up['time_stamp_diff'].sum()
        time_interval=int((dfa['time_stamp1'].max()-dfa['time_stamp1'].min()-time_leak)/timedelta(minutes=20))+1
        df_subset=np.array_split(dfa, time_interval)
        for k in range(time_interval):
            filtered_images=list()
            for x in range(len(df_subset[k])):
                if x==0:
                    filtered_images.append(df_subset[k].raw_image_filename.iloc[0])
                    x=x+1
                else:
                    imagename=df_subset[k].raw_image_filename.iloc[x]
                    original=cv2.imread(imagename)
                    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                    for m in filtered_images:
                        contrast=cv2.imread(m)
                        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                        ms,ssm=compare_images(original,contrast)
                        ms1.append(ssm)
                        if ssm > 0.75 or ms <1000:
                            break
                        else:
                            filtered_images.append(df_subset[k].raw_image_filename.iloc[x])
                            image_pth=imagename.split("/")[-1]
                            cv2.imwrite(store_dir+'/'+image_pth,original) #save the image in relative folder
                            break
            filter_im=pd.DataFrame(filtered_images,columns=['raw_image_filename'])
            filter_df=pd.concat([filter_df,filter_im])
        
    filter_df['current_box']=np.nan
    filter_df['current_annotation_file']=np.nan
    filter_df['current_person_count']=np.nan
    filter_df['current_posture']=np.nan
    filter_df.to_csv('pipeline_output/filtered_data_for_{}.csv'.format(u), mode='a', header=True,index=False)

