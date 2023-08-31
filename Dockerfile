# Use an official Python runtime as a parent image
FROM python:3.8 AS base

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN pip list

# Use the base as runtime
FROM base AS runtime
# Expose the port the app runs on
EXPOSE 5000
ENV SHELL=/bin/bash

# Run main.py when the container launches
CMD ["python", "app.py"]