from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import transformers

# Your existing code here...
state = None
current_steps = 25
attn_slicing_enabled = True
#mem_eff_attn_enabled = install_xformers
seed = None

model_id = 'stabilityai/stable-diffusion-2-1'

scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(
      model_id,
      revision="fp16",
      torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
      scheduler=scheduler
    ).to("cuda")
pipe.enable_attention_slicing()

pipe_i2i = None
pipe_upscale = None
pipe_inpaint = None
pipe_depth2img = None

#pipeline callbacks
def pipe_callback(step: int, timestep: int, latents: torch.FloatTensor):
    update_state(f"{step}/{current_steps} steps")

#generator for text to image and image to image
generator = torch.Generator('cuda').manual_seed(seed)

#upscaller
upscale_model_path = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'
upscaller = hub.load(upscale_model_path)
def upscale(img):
    result = upscaller(img)
    return result

#memory optimizer
def set_mem_optimizations(pipe):
    if attn_slicing_enabled:
        pipe.enable_attention_slicing()
    else:
        pipe.disable_attention_slicing()
    
    if mem_eff_attn_enabled:
        pipe.enable_xformers_memory_efficient_attention()

def update_state(new_state):
  global state
  state = new_state

def update_state_info(old_state):
    if state and state != old_state:
        return gr.update(value=state)

#to load text to image pipeline
def get_txt2img_pipe(scheduler):

    update_state("Loading Text to image model...")

    pipe = StableDiffusionPipeline.from_pretrained(
      model_id,
      revision="fp16" if torch.cuda.is_available() else "fp32",
      torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
      scheduler=scheduler
    ).to("cuda")
    pipe.enable_attention_slicing()
    set_mem_optimizations(pipe)
    pipe.to("cuda")
    return pipe

#to load image to image pipeline
def get_i2i_pipe(scheduler):
    
    update_state("Loading image to image model...")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
      model_id,
      revision="fp16" if torch.cuda.is_available() else "fp32",
      torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
      scheduler=scheduler,
      safety_checker=None,
      feature_extractor=None
    )
    set_mem_optimizations(pipe)
    pipe.to("cuda")
    return pipe

#if user selects text to image
def txt_to_img(prompt, n_images, neg_prompt, guidance, steps, width, height, generator, seed):

    pipe = get_txt2img_pipe(scheduler)

    result = pipe(
    prompt,
    num_images_per_prompt = n_images,
    negative_prompt = neg_prompt,
    num_inference_steps = int(steps),
    guidance_scale = guidance,
    width = width,
    height = height,
    generator = generator,
    callback=pipe_callback).images

    update_state(f"Done. Seed: {seed}")

    return result

#if user selects image to image
def img_to_img(prompt, n_images, neg_prompt, img, strength, guidance, steps, width, height, generator, seed):

    global pipe_i2i
    if pipe_i2i is None:
        pipe_i2i = get_i2i_pipe(scheduler)

    img = img['image']
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    result = pipe_i2i(
    prompt,
    num_images_per_prompt = n_images,
    negative_prompt = neg_prompt,
    image = img,
    num_inference_steps = int(steps),
    strength = strength,
    guidance_scale = guidance,
    # width = width,
    # height = height,
    generator = generator,
    callback=pipe_callback).images

    update_state(f"Done. Seed: {seed}")
        
    return result 

# Define your API endpoints
@csrf_exempt
def txt_to_img_api(request):
    if request.method == 'POST':
        # Extract request parameters
        prompt = request.POST.get('prompt')
        n_images = int(request.POST.get('n_images'))
        neg_prompt = request.POST.get('neg_prompt')
        guidance = float(request.POST.get('guidance'))
        steps = int(request.POST.get('steps'))
        width = int(request.POST.get('width'))
        height = int(request.POST.get('height'))
        seed = int(request.POST.get('seed'))

        # Call the text-to-image conversion function
        result = txt_to_img(prompt, n_images, neg_prompt, guidance, steps, width, height, generator, seed)

        # Prepare the response
        response = {
            'status': 'success',
            'result': result
        }
        return JsonResponse(response)
    else:
        response = {
            'status': 'error',
            'message': 'Invalid request method. Only POST requests are allowed.'
        }
        return JsonResponse(response, status=405)

@csrf_exempt
def img_to_img_api(request):
    if request.method == 'POST':
        # Extract request parameters
        prompt = request.POST.get('prompt')
        n_images = int(request.POST.get('n_images'))
        neg_prompt = request.POST.get('neg_prompt')
        img = request.FILES.get('image')  # Assumes the file field is named 'image'
        strength = float(request.POST.get('strength'))
        guidance = float(request.POST.get('guidance'))
        steps = int(request.POST.get('steps'))
        width = int(request.POST.get('width'))
        height = int(request.POST.get('height'))
        seed = int(request.POST.get('seed'))

        # Process the uploaded image
        img = Image.open(img)

        # Call the image-to-image conversion function
        result = img_to_img(prompt, n_images, neg_prompt, img, strength, guidance, steps, width, height, generator, seed)

        # Prepare the response
        response = {
            'status': 'success',
            'result': result
        }
        return JsonResponse(response)
    else:
        response = {
            'status': 'error',
            'message': 'Invalid request method. Only POST requests are allowed.'
        }
        return JsonResponse(response, status=405)