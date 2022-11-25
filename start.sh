#!/bin/bash

# Create the image of torch

echo "Checking if the Image already exists..."

if [[ "$(docker image inspect yolov7_detect_track_count:latest 2> /dev/null)" == [] ]];

then
    echo "The image doesn't exist. Building the Docker Image..."

    cd ./dependencies
    docker build -t yolov7_detect_track_count .
    cd ..

else
    echo "The image has already exists."
fi


# And then run
# 1. To allow the use of screen
echo "Setting up X Server to accept connections. Turning Access control off..."
xhost +

# 2. To run the image
echo "Runing the Docker Image in the current location -> Display: ON ; Web camera access: ON"
docker run --gpus all --rm -it -e DISPLAY=$DISPLAY -v  $PWD:/workspace -v /tmp/.X11-unix:/tmp/.X11-unix:rw --device="/dev/video0:/dev/video0"  yolov7_detect_track_count:latest
