#!/usr/bin/env bash

ip4=$(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)
jupyter notebook . --allow-root --ip=$ip4
