FROM ubuntu:22.04 as base
RUN apt-get update
RUN mkdir /home/csst && useradd -d /home/csst -s /usr/bin/bash csst
RUN apt-get install -y vim strace python3-dev wget libfftw3-dev build-essential libssl-dev git
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm -f get-pip.py

RUN pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pybind11 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install astropy -i https://pypi.tuna.tsinghua.edu.cn/simple

# install cmake
COPY cmake-3.30.0-rc4.tar.gz /home/csst/ 
RUN cd /home/csst && tar xzvf cmake-3.30.0-rc4.tar.gz 
RUN cd /home/csst/cmake-3.30.0-rc4 && ./configure && make -j16 && make install 
RUN rm -rf /home/csst/cmake-3.30.0-rc4.tar.gz /home/csst/cmake-3.30.0-rc4

#COPY  GalSim /home/csst/GalSim
#RUN cd /home/csst/GalSim && pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
