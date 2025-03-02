# Phi-4 Multimodal Instruct Server

## Self-Hosted OpenAI-Compatible Chat Completion Endpoint

This project enables you to self-host an OpenAI-compatible chat completion API endpoint powered by the [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) model. Deployed via a FastAPI server within a Docker container, this solution supports multimodal inputs—text, images, and audio—and leverages NVIDIA GPU acceleration for efficient inference.

Tested on AWS on G6 instance

## Features

- **Multimodal Capabilities**: Process text, image, and audio inputs seamlessly.
- **OpenAI Compatibility**: Integrate easily with the OpenAI Python client or any HTTP client.
- **GPU Support**: Utilize NVIDIA GPUs for enhanced performance.
- **Persistent Caching**: Store model weights in a Docker volume to avoid repeated downloads.
- **Scalable Design**: Handle multiple requests concurrently with Uvicorn workers.

## Prerequisites

Before you begin, ensure the following are installed on your system:

- **[Docker](https://docs.docker.com/get-docker/)**: For containerization.
- **[Docker Compose](https://docs.docker.com/compose/install/)**: For managing multi-container setups.
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)**: For GPU support within Docker.
- **A Compatible NVIDIA GPU**: Required for accelerated inference.

## Setup Instructions

Follow these steps to build and run the self-hosted endpoint.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Build the Docker Image

The provided `Dockerfile` sets up the environment with CUDA, Python, PyTorch, and required libraries. Build the image with:

```bash
docker build -t phi4-server .
```

This command creates an image named `phi4-server` with:
- CUDA 12.4.1 and Ubuntu 22.04 as the base.
- PyTorch, FastAPI, Uvicorn, and `flash-attn` installed.
- The server script (`server.py`) copied into the container.

### 3. Start the Container

Use the `docker-compose.yml` file to launch the service:

```bash
docker compose up -d
```

This command:
- Starts the `phi4-server` container in detached mode (`-d`).
- Maps port `8000` on the host to `8000` in the container.
- Mounts a volume (`hf_cache`) for caching model weights at `/app/hf_cache`.
- Configures GPU access via the NVIDIA Container Toolkit.

The API will be available at `http://localhost:8000`.

## Usage

Interact with the endpoint using the OpenAI Python client or any HTTP client. Below is an example demonstrating a request with text and an image.

### Example: Text and Image Request

```python
import openai
import base64

# Function to encode an image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Replace with the path to your image
image_path = "image.png"
image_base64 = encode_image(image_path)

# Initialize the OpenAI client
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="test",  # Use a real API key if authentication is enabled
)

# Send a multimodal request
response = client.chat.completions.create(
    model="Phi4",
    messages=[{
        "role": "user",
        "content": [{
            "type": "text",
            "text": "Describe this image."
        }, {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}"
            }
        }],
    }],
    max_completion_tokens=512,
)

print(response.choices[0].message.content)
```

### Supported Input Types

- **Text**: Add text entries with `"type": "text"` in the `"content"` list.
- **Image**: Include base64-encoded images or URLs with `"type": "image_url"`.
- **Audio**: Use `"type": "input_audio"` or `"type": "audio_url"` for audio inputs (if supported by `server.py`).

## Under the Hood

Here’s how the system operates behind the scenes.

### 1. **Docker Environment**

- **Base Image**: Built on `nvidia/cuda:12.4.1-devel-ubuntu22.04`, providing CUDA support.
- **Dependencies**: Installs Python 3, PyTorch with CUDA, FastAPI, Uvicorn, and `flash-attn` (version 2.7.4.post1) for optimized attention mechanisms.
- **Environment Variables**:
  - `HF_HOME=/app/hf_cache`: Sets the cache directory for Hugging Face model weights.
  - `PYTHONUNBUFFERED=1`: Ensures Python output is unbuffered for real-time logging.

### 2. **FastAPI Server**

- The `server.py` script (assumed to contain the API logic) is launched with Uvicorn:
  ```bash
  uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
  ```
- **Workers**: Runs 4 Uvicorn workers to process requests concurrently, leveraging the GPU.
- **Endpoint**: Exposes an OpenAI-compatible `/v1/chat/completions` endpoint (assumed based on compatibility).

### 3. **Model Processing**

- **Model**: Loads the `microsoft/Phi-4-multimodal-instruct` model from Hugging Face, configured to run on the GPU.
- **Multimodal Input**: The server processes text, images, and audio, passing them to the model for inference.
- **Optimization**: Uses `flash-attn` for efficient attention computation, reducing memory and compute overhead.

### 4. **Caching**

- Model weights are cached in `/app/hf_cache`, mapped to the `hf_cache` Docker volume.
- This persists weights across container restarts, eliminating the need for repeated downloads (saving time and bandwidth).

### 5. **GPU Utilization**

- The `docker-compose.yml` file reserves GPU resources:
  ```yaml
  deploy:
    resources:
      reservations:
        devices:
          - capabilities: [gpu]
  ```
- CUDA libraries within the container enable GPU-accelerated inference.

## Troubleshooting

- **GPU Access Issues**: Confirm the NVIDIA Container Toolkit is installed and the GPU is visible (`docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`).
- **Model Loading Errors**: Check if the `hf_cache` volume is populated. If not, ensure internet access and sufficient disk space.
- **API Failures**: Verify request formatting matches the expected structure (e.g., correct `"type"` fields for inputs).

## Contributing

Feel free to fork this repository, submit issues, or contribute pull requests to enhance the project!

## License

This project is released under the [MIT License](LICENSE).
