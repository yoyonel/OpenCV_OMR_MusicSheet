# https://hub.docker.com/r/chipgarner/opencv3-webcam/
FROM chipgarner/opencv3-webcam:python3

# ADD requirements.txt /requirements.txt

RUN apt-get update
# https://stackoverflow.com/questions/11114225/installing-scipy-and-numpy-using-pip
RUN apt-get -y install libblas-dev liblapack-dev libatlas-base-dev gfortran
RUN pip install scipy

RUN pip install numpy ffmpy matplotlib

# url: https://stackoverflow.com/questions/29274638/opencv-libdc1394-error-failed-to-initialize-libdc1394
RUN echo 'KERNEL=="raw1394", GROUP="video"' > /etc/udev/rules.d/raw1394.rules

WORKDIR /home

# url: http://www.camptocamp.com/actualite/utiliser-scripts-dentrypoint-docker-flexible/
COPY /docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

RUN apt-get update &&\
 	apt-get install -qqy \
 		git

# RUN apt-get update && apt-get install -qqy git zsh
# RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
# ENV TERM xterm-256color
# CMD ["/bin/zsh"]


RUN git clone --depth=1 https://github.com/Bash-it/bash-it.git ~/.bash_it &&\
	~/.bash_it/install.sh --silent
# ENV BASH_IT_THEME "powerline-multiline"