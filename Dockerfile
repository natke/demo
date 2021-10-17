FROM ubuntu:18.04

RUN apt-get update
RUN apt install -y python3-pip

# Python and pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN apt-get install -y python3-pip
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
RUN pip install --upgrade pip

RUN pip install torch==1.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install onnxruntime==1.7.0
RUN pip install transformers==3.0.2
RUN pip install onnx coloredlogs numpy


