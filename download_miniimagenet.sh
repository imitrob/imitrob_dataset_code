#!/bin/bash
wget -r -nH --cut-dirs=100 --no-check-certificate https://data.ciirc.cvut.cz/public/groups/incognite/Imitrob/mini-imagenet.zip
unzip mini-imagenet.zip
mv images miniimagenet
rm mini-imagenet.zip
