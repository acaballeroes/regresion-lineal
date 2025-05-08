# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (optional, if you plan to serve something)
EXPOSE 8000

# Run the script when the container launches
CMD ["python", "corrosion.py"]