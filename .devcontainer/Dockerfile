FROM mcr.microsoft.com/devcontainers/cpp:1-debian-11

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    openssh-server \
    openmpi-bin libopenmpi-dev \
    r-base-dev r-recommended qpdf

COPY tools/install.sh /tmp/
RUN chmod +x /tmp/install.sh

USER vscode
WORKDIR /home/vscode

RUN /tmp/install.sh rdep
