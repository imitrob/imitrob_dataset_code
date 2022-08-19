#!/usr/bin/env bash

docker build . -t trace-extractor
docker run -v $(realpath data):/root/tracer/data -it trace-extractor bash notebook.sh
