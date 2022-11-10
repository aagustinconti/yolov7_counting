#!/bin/bash

######## Host PC ########

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

# Instalation guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html


# Create the image of torch
cd ./dependencies_files
docker build docker build -t yolov7_detect_track_count
cd ..

# And then run
# 1. To allow the use of screen
xhost +

# 2.  To run the image
docker run --gpus all --rm -it -e DISPLAY=$DISPLAY -v  $PWD:/workspace -v /tmp/.X11-unix:/tmp/.X11-unix:rw --device="/dev/video0:/dev/video0"  yolov7_detect_track_count:latest
