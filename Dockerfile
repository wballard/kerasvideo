#base image provides CUDA support on Ubuntu 16.04
FROM nvidia/cuda:8.0-cudnn6-devel

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

#package updates to support conda
RUN apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz

#add on conda python and make sure it is in the path
RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet --output-document=anaconda.sh https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh && \
    /bin/bash /anaconda.sh -f -b -p $CONDA_DIR && \
    rm anaconda.sh

#conda installing python, then tensorflow and keras for deep learning
RUN pip install --upgrade pip && \
    pip install tensorflow==1.4.0 && \
    pip install keras==2.1.1 && \
    conda clean -yt

#all the code samples for the video series
VOLUME ["/src"]

#serve up a jupyter notebook 
WORKDIR /src
EXPOSE 8888
CMD jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root
