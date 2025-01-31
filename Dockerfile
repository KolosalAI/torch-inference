# Use an official lightweight Python image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /mod

# Copy the application files into the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "modules.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
