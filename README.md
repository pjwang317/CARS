# CARS

codes for the paper

Super-resolution of remotely sensed data using channel attention based deep learning approach

1. Data preparation:
cd data
run sampling_scale.py to generate the input pairs;
run filelist1.py to generate the filelist file.

2. For training:
 run CARS.py while set the mode='train'
 
3. For testing
  run CARS.py while set the mode='test'
  
note: the code is based on https://github.com/kuijiang0802/PECNN

If you find this is useful, please cite as:
Wang, Peijuan, Bulent Bayram, and Elif Sertel. "Super-resolution of remotely sensed data using channel attention based deep learning approach." International Journal of Remote Sensing 42, no. 16 (2021): 6050-6067.
