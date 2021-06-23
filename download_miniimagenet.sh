#!/bin/bash
mkdir ./miniimagenet
wget -r -nH --cut-dirs=100  https://data.ciirc.cvut.cz/public/groups/incognite/Imitrob/mini-imagenet.zip -P ./miniimagenet
unzip ./miniimagenet/mini-imagenet.zip
rm /miniimagenet/mini-imagenet.zip