import os
import torch
import requests
import time
from PIL import Image
from io import BytesIO
import json
import comfy.utils
import re
import base64
import folder_paths
from .utils import pil2tensor, tensor2pil


def get_config():
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
        with open(config_path, 'r') as f:  
            config = json.load(f)
        return config
    except:
        return {}

def save_config(config):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

class ComfyGeminiAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gemini-2.0-flash-exp-image", "placeholder": "Enter model name"}),
                "resolution": (
                    [
                        "512x512", 
                        "768x768", 
                        "1024x1024", 
                        "1280x1280", 
                        "1536x1536", 
                        "2048x2048",
                        "object_image size",
                        "subject_image size",
                        "scene_image size"
                    ], 
                    {"default": "1024x1024"}
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "timeout": ("INT", {"default": 120, "min": 10, "max": 600, "step": 10}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "object_image": ("IMAGE",),  
                "subject_image": ("IMAGE",),
                "scene_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("generated_images", "response", "image_url")
    FUNCTION = "process"
    CATEGORY = "ainewsto/Comfly_Gemini"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 120 

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def extract_image_urls(self, response_text):
        # Extract URLs from markdown image format: ![description](url)
        image_pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.findall(image_pattern, response_text)
        
        # If no markdown format found, extract raw URLs that look like image links
        if not matches:
            url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
            matches = re.findall(url_pattern, response_text)
            
        return matches if matches else []

    def resize_to_target_size(self, image, target_size):
        """Resize image to target size while preserving aspect ratio with padding"""
        # Convert PIL image to target size
        img_width, img_height = image.size
        target_width, target_height = target_size
        
        # Calculate the scaling factor to fit the image within the target size
        width_ratio = target_width / img_width
        height_ratio = target_height / img_height
        scale = min(width_ratio, height_ratio)
        
        # Calculate new dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize the image
        resized_img = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create a new blank image with the target size
        new_img = Image.new("RGB", (target_width, target_height), (255, 255, 255))
        
        # Calculate position to paste the resized image
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        # Paste the resized image
        new_img.paste(resized_img, (paste_x, paste_y))
        
        return new_img

    def parse_resolution(self, resolution_str):
        """Parse resolution string (e.g., '1024x1024') to width and height"""
        width, height = map(int, resolution_str.split('x'))
        return (width, height)

    def process(self, prompt, model, resolution, num_images, temperature, top_p, seed, timeout=120, 
                object_image=None, subject_image=None, scene_image=None, api_key=""):
        # Update API key if provided
        if api_key.strip():
            self.api_key = api_key
            config = get_config()
            config['api_key'] = api_key
            save_config(config)
            
        # Set the timeout value from user input
        self.timeout = timeout
        
        try:
            
            # Get current timestamp for formatting
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # Get target size based on resolution or input images
            target_size = None
            
            # Determine if we should use an input image's size
            if resolution == "object_image size" and object_image is not None:
                pil_image = tensor2pil(object_image)[0]
                target_size = pil_image.size
            elif resolution == "subject_image size" and subject_image is not None:
                pil_image = tensor2pil(subject_image)[0]
                target_size = pil_image.size
            elif resolution == "scene_image size" and scene_image is not None:
                pil_image = tensor2pil(scene_image)[0]
                target_size = pil_image.size
            else:
                # Use the specified resolution
                target_size = self.parse_resolution(resolution)
            
            # Check if we have image inputs
            has_images = object_image is not None or subject_image is not None or scene_image is not None
            
            # Prepare message content
            content = []
            
            # Build different prompts and content based on input type
            if has_images:
                # When we have image inputs, use the original prompt without enhancements
                content.append({"type": "text", "text": prompt})
                
                # Prepare descriptions for each image type
                image_descriptions = {
                    "object_image": "an object or item",
                    "subject_image": "a subject or character",
                    "scene_image": "a scene or environment"
                }
                
                # Add available images to content
                for image_var, image_tensor in [("object_image", object_image), 
                                             ("subject_image", subject_image), 
                                             ("scene_image", scene_image)]:
                    if image_tensor is not None:
                        # Convert tensor to PIL image
                        pil_image = tensor2pil(image_tensor)[0]
                        image_base64 = self.image_to_base64(pil_image)
                        content.append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        })
            else:
                # When we only have text input, add enhanced prompt
                # Get dimensions string from target_size
                dimensions = f"{target_size[0]}x{target_size[1]}"
                aspect_ratio = "1:1" if target_size[0] == target_size[1] else f"{target_size[0]}:{target_size[1]}"
                
                if num_images == 1:
                    enhanced_prompt = f"Generate a high-quality, detailed image with dimensions {dimensions} and aspect ratio {aspect_ratio}. Based on this description: {prompt}"
                else:
                    enhanced_prompt = f"Generate {num_images} DIFFERENT high-quality images with VARIED content, each with unique and distinct visual elements, all having the exact same dimensions of {dimensions} and aspect ratio {aspect_ratio}. Important: make sure each image has different content but maintains the same technical dimensions. Based on this description: {prompt}"
                
                content.append({"type": "text", "text": enhanced_prompt})
            
            # Create messages
            messages = [{
                "role": "user",
                "content": content
            }]
            
            # Create API payload
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed if seed > 0 else None,
                "max_tokens": 8192
            }
            
            # Make API request with progress bar
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)
            
            # Make API request with timeout
            try:
                response = requests.post(
                    "https://ai.comfly.chat/v1/chat/completions",
                    headers=self.get_headers(),
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.Timeout:
                raise TimeoutError(f"API request timed out after {self.timeout} seconds")
            except requests.exceptions.RequestException as e:
                raise Exception(f"API request failed: {str(e)}")
            
            pbar.update_absolute(40)
            
            # Extract response text
            response_text = result["choices"][0]["message"]["content"]
            
            # Format the response
            formatted_response = f"**User prompt**: {prompt}\n\n**Response** ({timestamp}):\n{response_text}"
            
            # Check if response contains image URLs
            image_urls = self.extract_image_urls(response_text)
            
            if image_urls:
                try:
                    # Process all images from URLs
                    images = []
                    first_image_url = ""  # Store the first image URL
                    
                    for i, url in enumerate(image_urls):
                        pbar.update_absolute(40 + (i+1) * 50 // len(image_urls))
                        
                        if i == 0:
                            first_image_url = url  # Save the first URL
                        
                        try:
                            img_response = requests.get(url, timeout=self.timeout)
                            img_response.raise_for_status()
                            pil_image = Image.open(BytesIO(img_response.content))
                            
                            # Resize the image to the target size if necessary
                            resized_image = self.resize_to_target_size(pil_image, target_size)
                            
                            # Convert to tensor
                            img_tensor = pil2tensor(resized_image)
                            images.append(img_tensor)
                            
                        except Exception as img_error:
                            print(f"Error processing image URL {i+1}: {str(img_error)}")
                            # Continue to next image if there's an error with this one
                            continue
                    
                    if images:
                        # If all images are the same size, we can use torch.cat
                        try:
                            combined_tensor = torch.cat(images, dim=0)
                        except RuntimeError:
                            # If images are different sizes, we'll need to handle them individually
                            print("Warning: Images have different sizes, returning first image")
                            combined_tensor = images[0] if images else None
                            
                        pbar.update_absolute(100)
                        return (combined_tensor, formatted_response, first_image_url)
                    else:
                        # If no images were successfully processed
                        raise Exception("No images could be processed successfully")
                    
                except Exception as e:
                    print(f"Error processing image URLs: {str(e)}")
            
            # Return appropriate response if no image URLs were found
            pbar.update_absolute(100)
            
            # Determine which image to return in case of no output images
            reference_image = None
            if object_image is not None:
                reference_image = object_image
            elif subject_image is not None:
                reference_image = subject_image
            elif scene_image is not None:
                reference_image = scene_image
                
            if reference_image is not None:
                # If any input image was provided, return the first available one with the text response
                return (reference_image, formatted_response, "")
            else:
                # Create a default blank image with the target size
                default_image = Image.new('RGB', target_size, color='white')
                default_tensor = pil2tensor(default_image)
                return (default_tensor, formatted_response, "")
            
        except TimeoutError as e:
            error_message = f"API timeout error: {str(e)}"
            print(error_message)
            return self.handle_error(object_image, subject_image, scene_image, error_message, resolution)
            
        except Exception as e:
            error_message = f"Error calling Gemini API: {str(e)}"
            print(error_message)
            return self.handle_error(object_image, subject_image, scene_image, error_message, resolution)
    
    def handle_error(self, object_image, subject_image, scene_image, error_message, resolution="1024x1024"):
        """Handle errors with appropriate image output"""
        # Return the first available image if any
        if object_image is not None:
            return (object_image, error_message, "")
        elif subject_image is not None:
            return (subject_image, error_message, "")
        elif scene_image is not None:
            return (scene_image, error_message, "")
        else:
            # Create an error image with the specified resolution
            # Handle custom resolution options
            if resolution in ["object_image size", "subject_image size", "scene_image size"]:
                target_size = (1024, 1024)  # Default if custom option selected but no image provided
            else:
                target_size = self.parse_resolution(resolution)
                
            default_image = Image.new('RGB', target_size, color='white')
            default_tensor = pil2tensor(default_image)
            return (default_tensor, error_message, "")

NODE_CLASS_MAPPINGS = {
    "ComfyGeminiAPI": ComfyGeminiAPI
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyGeminiAPI": "Gemini API"
}
