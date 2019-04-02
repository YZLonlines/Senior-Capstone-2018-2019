# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# update the conda
RUN conda update -n base -c defaults conda

# create a conda environment
RUN conda env create -f environment.yml

# Define environment variable
ENV PATH /opt/conda/envs/capstone/bin:$PATH
RUN /bin/bash -c "source activate capstone"

# install Java 
RUN apt-get update
RUN apt-get install -y openjdk-8-jdk

# set Java home
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
# CMD [ "source", "activate", "capstone" ]
