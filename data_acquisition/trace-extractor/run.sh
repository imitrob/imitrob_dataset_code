#!/usr/bin/env bash

docker build . -t trace-extractor
docker run -v $(realpath .):/root/tracer/ -it trace-extractor bash notebook.sh
