# Set up lmarspy environment

FROM docker.1ms.run/nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

SHELL ["bash", "--login", "-c"]

ENV CC=gcc
ENV CXX=g++
ENV FC=gfortran
ENV F77=gfortran

RUN apt-get update -y \
 && apt-get install -y gcc gfortran csh tcsh perl make vim \
    libhwloc-dev libfabric-dev libopenmpi-dev g++ cpp libxml2-dev libc6-dev wget

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

COPY /package/* /tmp

ENV HDF5_DIR=/opt/hdf5
ENV PATH=${HDF5_DIR}/bin:$PATH
ENV LD_LIBRARY_PATH=${HDF5_DIR}/lib:$LD_LIBRARY_PATH
RUN cd /tmp \
 && tar -xf hdf5-hdf5_1.14.5.tar.gz \
 && cd hdf5-hdf5_1.14.5 \
 && CC=mpicc CXX=mpicxx FC=mpif90 \
    ./configure --enable-hl --enable-fortran --enable-parallel --with-zlib \
                --enable-shared --enable-static --prefix=${HDF5_DIR} \
 && make -j8 && make install && ldconfig

ENV NETCDF_DIR=/opt/netcdf
ENV PATH=${NETCDF_DIR}/bin:$PATH
ENV LD_LIBRARY_PATH=${NETCDF_DIR}/lib:$LD_LIBRARY_PATH
RUN cd /tmp \
 && tar -xf netcdf-c-4.9.2.tar.gz \
 && cd netcdf-c-4.9.2 \
 && CC=mpicc CXX=mpicxx CPPFLAGS="-I${HDF5_DIR}/include" LDFLAGS="-L${HDF5_DIR}/lib" \
    ./configure --disable-byterange --enable-shared --enable-static --prefix=${NETCDF_DIR}  \
 && make -j8 && make install && ldconfig

RUN cd /tmp \
 && tar -xf netcdf-fortran-4.6.1.tar.gz \
 && cd netcdf-fortran-4.6.1/ \
 && CC=mpicc FC=mpif90 CFLAGS="-I${HDF5_DIR}/include -I${NETCDF_DIR}/include" LDFLAGS="-L${HDF5_DIR}/lib -L${NETCDF_DIR}/lib" \
    ./configure --enable-shared --enable-static --prefix=${NETCDF_DIR} \
 && make -j8 && make install && ldconfig

RUN cd /tmp \
 && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /opt/conda  \
 && rm Miniconda3-latest-Linux-x86_64.sh  \
 && /opt/conda/bin/conda init bash
ENV PATH=/opt/conda/bin:$PATH

RUN conda config --set custom_channels.conda-forge https://mirrors.cernet.edu.cn/anaconda/cloud/ \
 && python -m pip config set global.index-url https://mirrors.cernet.edu.cn/pypi/web/simple \
 && conda install -y python=3.10 numpy cython psutil pyyaml pytest colorama cftime numba scipy

 RUN python -m pip install mpi4py \
 && cd /tmp \
 && tar xvf netcdf4-python-1.7.2rel.tar.gz \
 && cd netcdf4-python-1.7.2rel \
 && CC=mpicc HDF5_DIR=/opt/hdf5 NETCDF4_DIR=/opt/netcdf MPI_INCDIR="${OMPI_HOME}/include" python setup.py install

RUN conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia

RUN cd / && rm -rf /tmp/* && pip cache purge && conda clean -a -y