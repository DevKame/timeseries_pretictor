### Dummy API README

#### Introduction
This is a simple and straightforward API designed to serve a dummy CSV file upon request. It's built using FastAPI and Docker for ease of deployment and usage.

#### Functionality
The API provides a single endpoint `/csv` to retrieve a CSV file. Upon making a POST request to this endpoint, the API responds with the dummy CSV file.

#### How to Use

1. **Build the Docker Image**: Navigate to the root directory of the cloned repository in your terminal and run the following command:
   ```
   docker compose build
   ```

2. **Run the Docker Container**: After building the Docker image, you can run the Docker container with the following command:
   ```
   docker compose up
   ```

3. **Access the API**: Once the container is running, you can access the API at `http://localhost:8000/csv`. You can use any HTTP client (e.g., curl, Postman) to make a POST request to this endpoint.

4. **Explore API Documentation**: FastAPI generates interactive API documentation automatically. You can explore and test the API endpoints by visiting `http://localhost:8000/docs` in your web browser. This documentation provides detailed information about the available endpoints and allows you to interactively make requests and view responses.