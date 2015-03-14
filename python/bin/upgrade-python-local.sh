#!/usr/bin/env bash

# install build tools 
yum install make automake gcc gcc-c++ kernel-devel git-core -y 
 
# install python 2.7 and change default python symlink 
yum install python27-devel -y 
rm /usr/bin/python
ln -s /usr/bin/python2.7 /usr/bin/python 
 
# point yum to the right version of python
sed -i s/python/python2.6/g /usr/bin/yum 
sed -i s/python2.6/python2.6/g /usr/bin/yum 
 
# install pip
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
 
# install matplotlib prereqs
yum install freetype-devel libpng-devel -y
 
# install python packages through pip
pip install numpy scipy six Pillow cython
pip install networkx matplotlib
pip install "ipython[notebook]" 
pip install seaborn boto
