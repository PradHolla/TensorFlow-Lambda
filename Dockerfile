# Pull the base image with python 3.8 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.9

# Install OS packages for Pillow-SIMD
RUN yum -y install tar gzip zlib freetype-devel \
    gcc \
    ghostscript \
    lcms2-devel \
    libffi-devel \
    libimagequant-devel \
    libjpeg-devel \
    libraqm-devel \
    libtiff-devel \
    libwebp-devel \
    make \
    openjpeg2-devel \
    rh-python36 \
    rh-python36-python-virtualenv \
    sudo \
    tcl-devel \
    tk-devel \
    tkinter \
    which \
    xorg-x11-server-Xvfb \
    zlib-devel \
    && yum clean all

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./

# Install the python requirements from requirements.txt
RUN pip install https://github.com/diyor28/tf-docker-m1/releases/download/v1.0.0/tensorflow-2.8.0-cp39-cp39-linux_aarch64.whl
# Replace Pillow with Pillow-SIMD to take advantage of AVX2
RUN pip install pillow && pip install numpy
# RUN pip uninstall -y pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
# Copy the earlier created app.py file to the container
COPY app.py ./

# Download ResNet50 and store it in a directory
RUN mkdir model
RUN curl -L https://ml-model-pnh.s3.amazonaws.com/gameClass.h5 -o ./model/gameClass.h5


# Set the CMD to your handler
CMD ["app.lambda_handler"]