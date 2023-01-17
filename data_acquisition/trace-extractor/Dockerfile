FROM ros:noetic-ros-base-buster

RUN apt-get update -y && apt-get upgrade -y

RUN apt-get install -y \
  graphviz \
  jupyter \
  libgl1-mesa-dev \
  python3-graphviz \
  python3-matplotlib \
  python3-numba \
  python3-pip \
  python3-sklearn \
  ros-noetic-geometry-tutorials \
  ros-noetic-ros-tutorials \
  ros-noetic-rosbash \
  ros-noetic-rqt-tf-tree \
  ros-noetic-rviz \
  wget \
  && :

RUN pip3 install spatialmath-python trimesh pyglet

WORKDIR /tmp

# ENV ANACONDA=Anaconda3-2022.05-Linux-x86_64
# RUN wget https://repo.anaconda.com/archive/$ANACONDA.sh
# RUN bash ./$ANACONDA.sh -b -p /root/anaconda
# COPY conda_env.txt conda_env.txt
# RUN /root/anaconda/bin/conda create --name tracer --file conda_env.txt

WORKDIR /root/tracer
# COPY . .

#RUN cat dot_bashrc >> /root/.bashrc
