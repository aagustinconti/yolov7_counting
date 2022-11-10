# Specify the parent image from which we build
FROM pytorch/pytorch:latest

# Copy the dependencies directory
COPY ./dependencies_files /dependencies_files_docker

# Set the working directory
WORKDIR /dependencies_files_docker

# Install dependencies
RUN pip install -r ./dependencies_files_docker/requirements.txt
RUN bash ./dependencies_files_docker/dependencies.sh

