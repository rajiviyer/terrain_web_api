# Use the python 3.10.13-bookworm container image
from python:3.10.13-bookworm

# Set the working directory to /app
WORKDIR /app

# Copy current directory contents into container at /app
ADD . /app

# Install the dependencies
RUN pip install -r requirements.txt

# Make port available to the world outside this container
EXPOSE 8502

HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health

# Run streamlit to start the application
CMD ["streamlit", "run", "wsgi.py"]
