from fastapi import FastAPI, Request
import requests
from PIL import Image
import base64
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from pydantic import BaseModel
import tempfile
import soundfile as sf
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Define model path
model_path = "microsoft/Phi-4-multimodal-instruct"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, image_processor_kwargs={'use_fast': True})

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype="auto", 
    trust_remote_code=True, 
    attn_implementation='flash_attention_2',
)
generation_config = GenerationConfig.from_pretrained(model_path)

def process_message_content(content_list):
    prompt_text = ""
    images = []
    audios = []
    image_count = 0
    audio_count = 0
    for item in content_list:
        if item["type"] == "text":
            prompt_text += item["text"]
        elif item["type"] == "image_url":
            image_url = item["image_url"]["url"]
            if image_url.startswith("data:image/"):
                # It's base64-encoded
                parts = image_url.split(",")
                if len(parts) == 2:
                    image_data = base64.b64decode(parts[1])
                    image = Image.open(BytesIO(image_data))
                else:
                    raise ValueError("Invalid base64 image URL")
            else:
                # It's a regular URL
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                image = Image.open(response.raw)
            image = image.convert('RGB')  # Ensure RGB mode
            images.append(image)
            logger.info(f"Loaded image from {image_url}, size: {image.size}, mode: {image.mode}")
            prompt_text += f"<|image_{image_count+1}|>"
            image_count += 1
        elif item["type"] == "input_audio":
            data = item["input_audio"]["data"]
            format = item["input_audio"]["format"]
            decoded_data = base64.b64decode(data)
            with tempfile.NamedTemporaryFile(suffix=f'.{format}') as temp_file:
                temp_file.write(decoded_data)
                temp_file.flush()
                audio, sample_rate = sf.read(temp_file.name)
                audios.append((audio, sample_rate))
            logger.info(f"Loaded audio from base64, shape: {audio.shape}, sample rate: {sample_rate}")
            prompt_text += f"<|audio_{audio_count+1}|>"
            audio_count += 1
        elif item["type"] == "audio_URL":
            url = item["audio_url"]["url"]
            with tempfile.NamedTemporaryFile() as temp_file:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file.flush()
                audio, sample_rate = sf.read(temp_file.name)
                audios.append((audio, sample_rate))
            logger.info(f"Loaded audio from {url}, shape: {audio.shape}, sample rate: {sample_rate}")
            prompt_text += f"<|audio_{audio_count+1}|>"
            audio_count += 1
    return prompt_text, images, audios


def generate_response(prompt_text, images, audios):
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    full_prompt = f"{user_prompt}{prompt_text}{prompt_suffix}{assistant_prompt}"
    
    logger.info(f"Prompt text: {prompt_text}")
    logger.info(f"Number of images: {len(images)}")
    if len(images) == 0:
        logger.info("No images provided")
        images = None
    logger.info(f"Number of audios: {len(audios)}")
    if len(audios) == 0:
        logger.info("No audios provided")
        audios = None
    
    try:
        inputs = processor(text=full_prompt, images=images, audios=audios, return_tensors='pt').to(model.device)
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        raise

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    max_completion_tokens: int

class ChatCompletionResponse(BaseModel):
    choices: list[dict]

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Extract the last user message
    user_message = None
    for message in request.messages:
        if message["role"] == "user":
            user_message = message
    if not user_message:
        return {"error": "No user message found"}
    
    content_list = user_message["content"]
    prompt_text, images, audios = process_message_content(content_list)
    
    response_text = generate_response(prompt_text, images, audios)
    
    # Format the response
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ]
    }
    
    return response

@app.get("/health")
async def health():
    return {"status": "OK"}