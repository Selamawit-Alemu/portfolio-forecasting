# Use a slim, official Python image for a smaller footprint
FROM python:3.12-slim-bookworm

# Set the working directory within the container
WORKDIR /app

# Copy only the requirements file first to optimize Docker layer caching.
# This ensures that if only your code changes, the dependencies aren't reinstalled.
COPY requirements.txt .

# Install Python dependencies.
# --no-cache-dir: Prevents pip from storing cache, reducing image size.
# --compile: Compiles Python source files to bytecode, potentially speeding up startup.
# --prefer-binary: Prefers pre-compiled wheels, which can make installations faster.
RUN pip install --no-cache-dir --compile --prefer-binary -r requirements.txt

# Copy the rest of your application's source code into the container
COPY . .

# Expose the default Jupyter Notebook port
EXPOSE 8888

# Command to run Jupyter Notebook.
# --ip=0.0.0.0: Makes Jupyter accessible from outside the container.
# --port=8888: Specifies the port Jupyter will listen on.
# --no-browser: Prevents Jupyter from trying to open a browser inside the container.
# --allow-root: Allows running Jupyter as root (useful in some container environments,
#               but consider running as a non-root user for better security if possible).
# --NotebookApp.token='': Disables token authentication for easier access in development.
#                         For production, you should secure your Jupyter instance.
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]