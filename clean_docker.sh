#!/bin/bash
docker stats --no-stream --format "{{.Container}}: {{.PIDs}}" | while read line; do
    container_id=$(echo $line | cut -d':' -f1)
    pids=$(echo $line | cut -d':' -f2 | tr -d ' ')
    if [ "$pids" -lt 7 ]; then
        docker rm -f $container_id
    fi
done
