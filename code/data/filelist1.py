import os

# DIR='D:\\Dataset\\DIV2K\\DIV2K_train_LR\\X4\\'
DIR = 'D:\\super-resolution\\MFSR\\SNet7\\data\\test\\'
all_name=os.listdir(DIR)
file_object = open('filelist_test_SNet7.txt', 'w')
file_list=''
for f in all_name:
    if os.path.isdir(os.path.join(DIR,f)):
        #file_list+=os.path.join(DIR,f)+'\n'
        file_list+=DIR+'/'+f+'\n'
        #print(f)
        print (file_list)
file_object.write(file_list)
file_object.close( )