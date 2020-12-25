FROM gcc:cmake
RUN apt-get update && \
apt-get install -y ssh \
gdb \
tar \
python && \
apt-get clean

RUN useradd -m user && yes password | passwd user

RUN mkdir /var/run/sshd
