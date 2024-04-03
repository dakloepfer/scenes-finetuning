#!/bin/bash -l

eurocmav_dir="./euroc-mav"

mkdir $eurocmav_dir/index
cp ./train_list.txt $eurocmav_dir/index/train_list.txt
cp ./val_list.txt $eurocmav_dir/index/val_list.txt
cp ./test_list.txt $eurocmav_dir/index/test_list.txt

echo "Undistorting images..."
python eurocmav-undistort.py --data_path $eurocmav_dir

echo "Computing poses..."
python eurocmav-compute_poses.py --data_path $eurocmav_dir

echo "Making image pairs..."
mkdir $eurocmav_dir/index/trainval
mkdir $eurocmav_dir/index/test
python eurocmav-make_npz_files.py --data_path $eurocmav_dir 

echo "Finished preparing the Euroc MAV dataset!"