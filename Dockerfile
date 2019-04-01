# Use an official Python runtime as a parent image
FROM python:3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r environment.yml

# RUN cp -R /usr/local/lib/python3.7/site-packages/PyQt5/Qt/plugins/platforms /app

# Define environment variable
ENV NAME World

# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update
# RUN apt-get install -y software-properties-common
# RUN add-apt-repository -y ppa:openjdk-r/ppa

RUN apt-get install -y openjdk-8-jdk

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# RUN export QT_DEBUG_PLUGINS=1

# RUN apt-get install libxkbcommon-x11-dev

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "src/Driver.py", "BFSIZE HDRSIZE NODETYPE NODESTATE METADATASIZE", "RandomForest"]
