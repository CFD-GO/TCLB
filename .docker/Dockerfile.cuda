FROM nvidia/cuda:8.0-devel-ubuntu16.04

ENV docker_tag=CUDA

RUN useradd docker \
	&& mkdir /home/docker \
	&& chown docker:docker /home/docker \
	&& addgroup docker staff

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		ed \
		less \
		locales \
		vim-tiny \
		wget \
		ca-certificates \
		fonts-texgyre \
		git \
		software-properties-common \
		autoconf \
		ssh \
		readline-common \
		joe \
	&& rm -rf /var/lib/apt/lists/*

COPY tools /tools

RUN /tools/install.sh r
RUN /tools/install.sh openmpi

COPY --chown=docker:docker .docker/bashrc /home/docker/.bashrc

WORKDIR /home/docker
USER docker

RUN /tools/install.sh rdep

CMD ["bash"]
