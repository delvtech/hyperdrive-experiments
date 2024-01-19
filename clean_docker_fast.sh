#!/bin/bash
num_cores=$(nproc)
docker stats --no-stream --format "{{.Container}}: {{.PIDs}}" | \
    awk -F ": " '{if ($2 < 7) print $1}' | \
    xargs -r -P 7 -I {} docker rm -f {}
