# Use an official Python runtime as a parent image
FROM python:3.10.4-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run gunicorn server when the container launches
CMD ["gunicorn", "-w", "1", "-b", ":5000", "-t", "300", "app:app"]
