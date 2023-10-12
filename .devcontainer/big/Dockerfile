FROM mcr.microsoft.com/devcontainers/cpp:1-debian-11

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    openssh-server \
    python3-pip \
    libxml2 libxml2-dev \
    openmpi-bin libopenmpi-dev \
    r-base-dev r-recommended qpdf \
    libgl1-mesa-glx

COPY tools/install.sh /tmp/
RUN chmod +x /tmp/install.sh

USER vscode
WORKDIR /home/vscode

RUN /tmp/install.sh rdep rinside reticulate --rpackage languageserver --rpackage png
RUN pip3 install vtk
ENV RETICULATE_PYTHON=/usr/bin/python3
