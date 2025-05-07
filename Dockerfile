# start from lightning installation
FROM python:3.12.4-bullseye

# create a working dicrectory named "project"
WORKDIR /project

# copy the requirements of the project into the container
COPY requirements.txt requirements.txt

# install the requirements of the project
RUN pip install -r requirements.txt

# copy the project code the workind directory of the container
COPY . .

# open a bash
CMD ["/bin/bash"]
