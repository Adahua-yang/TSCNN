#! /bin/bash

root_path="./"
download_url="http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar"
rgb_root=$root_path"/i-LIDS-VID/sequences"
flow_root=$root_path"/i-LIDS-VID-EpicFlow/sequences"

# make dirs.
wget $download_url
tar -xvf iLIDS-VID.tar
mkdir -p $flow_root

# extract epic flow
git clone git@github.com:zyoohv/epicflow-python3.git
python epicflow-python3/abstract_epic.py --rgb_root "$rgb_root" --flow_root "$flow_root"

# train our network
cd base_model
./run