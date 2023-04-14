# Specify the parent image from which we build
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Copy the dependencies directory
COPY . /dependencies_files_docker

# Set the working directory
WORKDIR /dependencies_files_docker

# Install dependencies
RUN pip install -r requirements.txt
RUN chmod +x dependencies.sh
RUN bash dependencies.sh
RUN mv /opt/conda/lib/python3.10/site-packages/torch/nn/modules/upsampling.py /opt/conda/lib/python3.10/site-packages/torch/nn/modules/upsampling_wrong.py
RUN cp -f ./upsampling.py /opt/conda/lib/python3.10/site-packages/torch/nn/modules/upsampling.py

# To go to the working directory
WORKDIR /workspace

