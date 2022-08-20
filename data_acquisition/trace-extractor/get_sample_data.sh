#!/bin/bash
wget -r -nH --cut-dirs=100  https://data.ciirc.cvut.cz/public/groups/incognite/trace-extractor/trace-extractor-data-main.zip
unzip trace-extractor-data-main.zip
mv trace-extractor-data-main data
rm trace-extractor-data-main.zip
