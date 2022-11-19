#!/bin/bash

######## Host PC ########

# Pre run
# 1. sudo chmod +x start.sh

# Do the complete instalation of Docker.

# 1. Install: https://docs.docker.com/engine/install/ubuntu/
# 2. Post-install: https://docs.docker.com/engine/install/linux-postinstall/


# Install Nvidia Drivers.

# Instalation:
# 1. ubuntu-drivers devices  (To know which driver is recommended)
# 2. sudo ubuntu-drivers autoinstall (To automatically install the recommended driver)
# 3. reboot # We need to reboot the system after the installation
# Checks:
# 1. nvidia-smi (Command to check if the driver was installed correctly: The output must be a list of GPU's and a list processes running on it)
# 2. sudo apt install nvtop (Program to check the GPU usage)
# 3. nvtop (To run the nvtop)

# Pull nvidia-gpu image

# 1. Instalation guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html


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
