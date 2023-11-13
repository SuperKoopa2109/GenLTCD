
# Make data directory
RUN mkdir -p /data

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 rsync -y
RUN apt-get -y install \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-setuptools \
    python3-pip \
    nano \
    vim \
    zsh

# Install requirements.txt
RUN pip install -r /install/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# save some space
RUN rm -rf /root/.cache/pip

# install oh-my-zsh
RUN git clone "https://github.com/robbyrussell/oh-my-zsh.git" "${HOME}/.oh-my-zsh"
RUN cp "${HOME}/.oh-my-zsh/templates/zshrc.zsh-template" "${HOME}/.zshrc"