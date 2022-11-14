# Specify the parent image from which we build
FROM pytorch/pytorch:latest

# Copy the dependencies directory
COPY . /dependencies_files_docker

# Set the working directory
WORKDIR /dependencies_files_docker

# Install dependencies
RUN pip install -r requirements.txt
RUN chmod +x dependencies.sh
RUN bash dependencies.sh
RUN cp -f ./upsampling.py /opt/conda/lib/python3.7/site-packages/torch/nn/modules/upsampling.py

# To go to the working directory
WORKDIR /workspace

