from flask import Flask, render_template, request, send_file
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from bs4 import BeautifulSoup
import time
import io
from rembg import remove
from moviepy.video.VideoClip import ColorClip
import re
from moviepy.video.fx.resize import resize
import traceback
from anthropic import Anthropic
import cv2
import gc
import base64

app = Flask(__name__)
load_dotenv(override=True)

# Initialize API clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

# Add D-ID API credentials
DID_API_URL = os.getenv('DID_API_URL')
DID_API_KEY = os.getenv('DID_API_KEY')

# Initialize Anthropic client at the top with other API clients
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def fetch_images_from_google(query):
    """Fetch images using Google Custom Search API"""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        print("Error: Google API key or CSE ID is not set")
        return []
    
    # Extract the main product name (first few words only)
    main_product = ' '.join(query.split()[:3])  # Take first 3 words
    
    search_terms = [
        main_product,
        f"{main_product} product",
        f"{main_product} review",
        f"{main_product} features",
        f"{main_product} lifestyle"
    ]
    
    images = []
    used_links = set()  # Track unique image URLs
    
    for term in search_terms:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': term,
            'cx': GOOGLE_CSE_ID,
            'key': GOOGLE_API_KEY,
            'searchType': 'image',
            'num': 10,
            'imgSize': 'large',
            'imgType': 'photo',
            'safe': 'active'
        }
        
        try:
            response = requests.get(url, params=params)
            print(f"Google API Response Status for '{term}': {response.status_code}")
            
            if response.status_code == 200:
                results = response.json().get('items', [])
                for item in results:
                    image_url = item['link']
                    # Only add unique images and filter out unwanted domains
                    if (image_url not in used_links and 
                        not any(domain in image_url.lower() for domain in [
                            'pinterest', 'facebook', 'instagram', 'twitter',
                            'tiktok', 'youtube', 'ebay'
                        ])):
                        try:
                            # Test if image is accessible
                            img_response = requests.head(image_url, timeout=2)
                            if img_response.status_code == 200:
                                images.append(image_url)
                                used_links.add(image_url)
                                print(f"Added image: {image_url[:100]}...")
                        except:
                            print(f"Skipped inaccessible image: {image_url[:100]}...")
            else:
                print(f"Error response from Google: {response.text}")
        except Exception as e:
            print(f"Error fetching images for term '{term}': {str(e)}")
        
        if len(images) >= 10:  # Stop if we have enough unique images
            break
    
    print(f"Found {len(images)} unique images")
    return images[:10]  # Return up to 10 unique images

def generate_script(product_name, product_description, video_length):
    """Generate video script using OpenAI with strict word count based on time"""
    # Calculate target word count (average speaking rate is ~150 words per minute)
    target_words = int((int(video_length) / 60) * 150)
    
    prompt = f"""Write a {video_length} second promotional video script for this product:
    Product: {product_name}
    Description: {product_description}
    
    CRITICAL: The script MUST be exactly {target_words} words to fit in {video_length} seconds.
    
    Write ONLY the speaking parts. No labels, no directions - just the exact words to be spoken.
    Make it engaging and natural, like someone talking to a friend.
    Include an opening hook, key features, and end with a call to action."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        
        script = response.choices[0].message.content.strip()
        
        # Clean up formatting
        script = script.replace('SCRIPT:', '').replace('VISUALS:', '')
        script = script.replace('Speaker:', '').replace('Narrator:', '')
        script = re.sub(r'\([^)]*\)', '', script)
        script = re.sub(r'\[[^\]]*\]', '', script)
        script = '\n'.join(line for line in script.split('\n') 
                          if not line.strip().startswith(('(', '[', '*', '-')))
        
        return script.strip(), ""
        
    except Exception as e:
        print(f"Error generating script: {str(e)}")
        raise e

def get_presenter_prompt(product_name=None, product_description=None):
    """Generate a prompt for DALL-E that will create a context-appropriate presenter image"""
    
    # Analyze product context for gender appropriateness
    feminine_keywords = ['women', 'woman', 'feminine', 'girl', 'female', 'makeup', 'beauty', 'skincare', 
                        'cosmetic', 'purse', 'handbag', 'dress', 'pregnancy', 'maternity']
    masculine_keywords = ['men', 'man', 'masculine', 'guy', 'male', 'beard', 'shaving', 'suit', 
                        'tie', 'aftershave', 'cologne']
    
    # Convert to lower case for matching
    product_text = (product_name + " " + product_description).lower()
    
    # Determine presenter gender based on product context
    is_feminine = any(keyword in product_text for keyword in feminine_keywords)
    is_masculine = any(keyword in product_text for keyword in masculine_keywords)
    
    # Default to professional woman if no clear gender context
    if is_masculine and not is_feminine:
        presenter_type = "Professional male presenter"
    else:
        presenter_type = "Professional female presenter"
    
    base_prompt = f"""{presenter_type}, direct front view portrait, shoulders up, 
    looking directly at camera, natural smile, casual attire, studio lighting, 
    solid black background, 8K quality, photorealistic.

    CRITICAL REQUIREMENTS:
    - Only one person in the image
    - Direct front-facing pose only
    - Eyes looking straight at camera
    - Head centered in frame, make sure it is not cut out of frame. Head and shoulds fully in frame. 
    - Casual attire
    - Shoulders and head only
    - Solid black background
    - Well-lit face with studio lighting
    - Sharp, clear facial features
    - Professional corporate headshot style
    - No artistic effects
    - No side angles
    - No dramatic lighting
    - No creative poses
    - No props or products in the image, only the presenter
    """

    if product_name and product_description:
        # Add product context while maintaining core requirements
        base_prompt += f"\nPresenter should appear knowledgeable about {product_name}, maintaining professional corporate style."
        
        # Add specific styling notes based on product context
        if is_masculine:
            base_prompt += "\nWell-groomed male presenter in casual attire, clean-shaven or neat beard."
        elif is_feminine:
            base_prompt += "\nWell-groomed female presenter in casual attire, natural makeup."
    
    return base_prompt

def process_presenter_image(presenter_image):
    """Process presenter image with fully transparent background"""
    try:
        with Image.open(presenter_image) as img:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Super aggressive background removal
            img_no_bg = remove(
                img,
                alpha_matting=True,
                alpha_matting_foreground_threshold=250,
                alpha_matting_background_threshold=5,
                alpha_matting_erode_size=20
            )
            
            # Convert to numpy array for processing
            img_array = np.array(img_no_bg)
            
            # Create binary mask for alpha channel
            # Anything not completely opaque becomes completely transparent
            alpha = img_array[:, :, 3]
            alpha = np.where(alpha > 240, 255, 0)  # Very strict threshold
            img_array[:, :, 3] = alpha
            
            # Zero out RGB values where alpha is 0
            img_array[alpha == 0] = [0, 0, 0, 0]
            
            # Convert back to PIL Image
            output = Image.fromarray(img_array, 'RGBA')
            
            # Save as PNG with maximum compression
            processed_path = "frames/processed_presenter.png"
            output.save(processed_path, 'PNG', optimize=True)
            
            return processed_path
            
    except Exception as e:
        print(f"Error processing presenter image: {str(e)}")
        return None

def generate_ai_presenter(product_name=None, product_description=None):
    """Generate a presenter image using DALL-E based on product context"""
    try:
        print("Generating AI presenter image...")
        
        # Get contextual prompt based on product
        dalle_prompt = get_presenter_prompt(product_name, product_description) if product_name else "Casual presenter against plain background, high quality professional headshot, facing forward, neutral expression, shoulders up, casual attire, studio lighting, 4K"
        
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=dalle_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        # Get the image URL
        image_url = response.data[0].url
        
        # Download the image
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            # Save the image
            os.makedirs('frames', exist_ok=True)
            presenter_path = "frames/ai_presenter.jpg"
            with open(presenter_path, 'wb') as f:
                f.write(image_response.content)
            return presenter_path
        else:
            raise Exception("Failed to download generated image")
            
    except Exception as e:
        print(f"Error generating AI presenter: {str(e)}")
        return None

def create_talking_avatar(presenter_image, script, product_name=None, product_description=None):
    """Create a talking avatar video using D-ID API with gender-appropriate voice"""
    try:
        # First determine if it's an AI-generated or uploaded image
        is_uploaded = not str(presenter_image).startswith('frames/ai_presenter')
        print(f"Image path: {presenter_image}")
        print(f"Is uploaded: {is_uploaded}")
        
        # If uploaded, use OpenAI Vision to detect gender
        if is_uploaded:
            try:
                # Convert image to base64 for OpenAI Vision API
                with open(presenter_image, 'rb') as img_file:
                    image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                
                response = openai_client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Is this a photo of a woman or a man? Reply with only one word: 'woman' or 'man'."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=10
                )
                
                gender = response.choices[0].message.content.strip().lower()
                print(f"Detected gender from image: {gender}")
                
                # Force female voice for woman
                if gender == "woman":
                    voice_id = "en-US-AvaMultilingualNeural"
                    print("Using female voice based on image detection")
                else:
                    voice_id = "en-US-BrianNeural"
                    print("Using male voice based on image detection")
                    
            except Exception as e:
                print(f"Error in gender detection: {str(e)}")
                print("Falling back to female voice")
                voice_id = "en-US-AvaMultilingualNeural"  # Default to female voice if detection fails
        else:
            # For AI-generated images, use product context
            feminine_keywords = ['women', 'woman', 'feminine', 'beauty', 'skincare', 'makeup', 'cosmetic']
            is_feminine = any(keyword in (product_name or '').lower() + (product_description or '').lower() 
                            for keyword in feminine_keywords)
            voice_id = "en-US-AvaMultilingualNeural" if is_feminine else "en-US-BrianNeural"
            print(f"Using {'female' if is_feminine else 'male'} voice based on product context")

        print(f"Final voice selection: {voice_id}")
        
        # Process and upload image
        processed_image = process_presenter_image(presenter_image)
        if not processed_image:
            raise Exception("Failed to process presenter image")
            
        # Upload image to D-ID
        print("Uploading image to D-ID...")
        with open(processed_image, 'rb') as img_file:
            files = {
                'image': ('presenter.jpg', img_file, 'image/jpeg')
            }
            
            upload_headers = {
                "Authorization": f"Basic {DID_API_KEY}"
            }
            
            upload_response = requests.post(
                f"{DID_API_URL}/images",
                headers=upload_headers,
                files=files
            )
        
        if upload_response.status_code not in [200, 201]:
            raise Exception(f"Failed to upload image: {upload_response.text}")
            
        image_url = upload_response.json()['url']
        print(f"Image uploaded successfully. URL: {image_url}")

        # Create the talk with D-ID's audio
        payload = {
            "script": {
                "type": "text",
                "input": script,
                "provider": {
                    "type": "microsoft",
                    "voice_id": voice_id
                }
            },
            "config": {
                "fluent": True,
                "pad_audio": 0,
                "result_format": "mp4",
                "remove_background": True,
                "transparent_background": True,
                "stitch": True
            },
            "source_url": image_url
        }

        talk_headers = {
            "Authorization": f"Basic {DID_API_KEY}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        
        response = requests.post(
            f"{DID_API_URL}/talks",
            headers=talk_headers,
            json=payload
        )
        
        if response.status_code != 201:
            raise Exception(f"Failed to create talk: {response.text}")
            
        talk_id = response.json()['id']
        print(f"Talk created with ID: {talk_id}")
        
        # Wait for processing
        max_retries = 60
        for i in range(max_retries):
            status_response = requests.get(
                f"{DID_API_URL}/talks/{talk_id}",
                headers={"Authorization": f"Basic {DID_API_KEY}"}
            )
            
            status = status_response.json().get('status')
            print(f"Processing status ({i+1}/{max_retries}): {status}")
            
            if status == 'done':
                result_url = status_response.json()['result_url']
                print(f"Success! Downloading from: {result_url}")
                
                video_response = requests.get(result_url)
                avatar_path = "frames/avatar.mp4"
                
                with open(avatar_path, 'wb') as f:
                    f.write(video_response.content)
                
                return avatar_path
                
            elif status == 'error':
                error_message = status_response.json().get('error', 'Unknown error')
                raise Exception(f"D-ID processing failed: {error_message}")
                
            time.sleep(2)
            
        raise Exception("Timeout waiting for video processing")
        
    except Exception as e:
        print(f"Error in create_talking_avatar: {str(e)}")
        traceback.print_exc()
        return None

def process_avatar_video(avatar_path):
    """Simplified function that just returns the original video path"""
    print("Background removal disabled - using original video")
    return avatar_path

def create_video(images, script, presenter_image=None, product_name=None, product_description=None):
    try:
        print("=== Starting Video Creation Process ===")
        
        os.makedirs('frames', exist_ok=True)
        os.makedirs('static', exist_ok=True)
        
        # Process frames for main content
        frames = []
        target_size = (1920, 1080)
        
        for idx, img_url in enumerate(images):
            try:
                print(f"Processing image {idx + 1}/{len(images)}: {img_url[:50]}...")
                
                # Download and process product images
                img_response = requests.get(img_url, timeout=10)
                if img_response.status_code == 200:
                    temp_img_path = f"frames/frame_{idx}.jpg"
                    with open(temp_img_path, "wb") as f:
                        f.write(img_response.content)
                    
                    # Process image with PIL
                    img = Image.open(temp_img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Calculate resize dimensions
                    aspect_ratio = img.width / img.height
                    if aspect_ratio > target_size[0] / target_size[1]:
                        new_size = (target_size[0], int(target_size[0] / aspect_ratio))
                    else:
                        new_size = (int(target_size[1] * aspect_ratio), target_size[1])
                    
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Create white background
                    background = Image.new('RGB', target_size, (255, 255, 255))
                    
                    # Paste image in center
                    paste_x = (target_size[0] - new_size[0]) // 2
                    paste_y = (target_size[1] - new_size[1]) // 2
                    background.paste(img, (paste_x, paste_y))
                    
                    frames.append(np.array(background))
                    
            except Exception as e:
                print(f"Error processing image {idx}: {str(e)}")
                continue
        
        if not frames:
            raise Exception("No valid frames were created")
        
        # Load and process avatar video
        if presenter_image:
            avatar_path = create_talking_avatar(presenter_image, script, product_name, product_description)
            if avatar_path and os.path.exists(avatar_path):
                try:
                    print("Loading avatar clip with audio...")
                    avatar_clip = VideoFileClip(avatar_path)
                    avatar_duration = avatar_clip.duration
                    
                    # Calculate duration for each image based on avatar duration
                    image_duration = avatar_duration / len(frames)
                    print(f"Setting each image duration to: {image_duration} seconds")
                    
                    print("Creating main clip...")
                    main_clip = ImageSequenceClip(frames, durations=[image_duration] * len(frames))
                    
                    print("Creating mask for avatar...")
                    def create_mask(frame):
                        if len(frame.shape) == 4:
                            frame = frame[:,:,:3]
                        # Less aggressive thresholds
                        is_light = np.mean(frame, axis=2) > 245
                        is_white = np.all(frame > [252, 252, 252], axis=2)
                        remove_mask = is_light & is_white
                        keep_mask = ~remove_mask
                        return keep_mask.astype('float32')
                    
                    # Apply mask to each frame
                    avatar_frames = []
                    for frame in avatar_clip.iter_frames():
                        mask = create_mask(frame)
                        rgba_frame = np.zeros((frame.shape[0], frame.shape[1], 4))
                        rgba_frame[:,:,:3] = frame[:,:,:3]
                        rgba_frame[:,:,3] = mask * 255
                        avatar_frames.append(rgba_frame)
                    
                    print("Creating masked avatar clip...")
                    masked_avatar = ImageSequenceClip(avatar_frames, fps=avatar_clip.fps)
                    
                    print("Resizing avatar...")
                    avatar_width = int(target_size[0] * 0.30)  # 20% of screen width
                    masked_avatar = resize(masked_avatar, width=avatar_width)
                    
                    # Position avatar in bottom right with padding
                    avatar_x = target_size[0] - avatar_width - 50
                    avatar_y = target_size[1] - masked_avatar.h - 50
                    
                    print("Creating composite clip...")
                    final_clip = CompositeVideoClip(
                        [
                            main_clip,
                            masked_avatar.set_position((avatar_x, avatar_y))
                        ],
                        size=target_size
                    ).set_audio(avatar_clip.audio)  # Explicitly set the audio from avatar_clip
                    
                    print("Writing final video...")
                    output_path = "static/generated_video.mp4"
                    final_clip.write_videofile(
                        output_path,
                        fps=30,
                        codec='libx264',
                        audio_codec='aac',
                        audio_bitrate="192k",
                        bitrate="8000k"
                    )
                    
                    print("Cleaning up clips...")
                    main_clip.close()
                    avatar_clip.close()
                    masked_avatar.close()
                    final_clip.close()
                    
                    return output_path
                    
                except Exception as e:
                    print(f"Error in video composition: {str(e)}")
                    traceback.print_exc()
                    return None
                    
    except Exception as e:
        print(f"Error in create_video: {str(e)}")
        traceback.print_exc()
        return None

def create_basic_video(frames, video_or_audio_file):
    """Fallback function to create basic video with audio"""
    try:
        # Load the audio from either video or audio file
        if video_or_audio_file.endswith('.mp4'):
            audio_clip = VideoFileClip(video_or_audio_file).audio
        else:
            audio_clip = AudioFileClip(video_or_audio_file)
        
        # Create video clip and set its duration to match audio
        clip = ImageSequenceClip(frames, durations=[3] * len(frames))
        clip = clip.set_duration(audio_clip.duration)
        
        # Set the audio
        final_clip = clip.set_audio(audio_clip)
        
        output_path = "static/generated_video.mp4"
        final_clip.write_videofile(
            output_path,
            fps=30,
            codec='libx264',
            audio_codec='aac',
            audio_bitrate="192k"
        )
        
        # Clean up
        clip.close()
        audio_clip.close()
        final_clip.close()
        
        return output_path
    except Exception as e:
        print(f"Error creating basic video: {str(e)}")
        raise e

def generate_voiceover(script):
    """Generate voiceover using OpenAI TTS"""
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=script
    )
    
    # Save the audio file
    audio_file = "temp_audio.mp3"
    with open(audio_file, 'wb') as f:
        for chunk in response.iter_bytes():
            f.write(chunk)
    return audio_file

def get_product_info(amazon_link):
    """Fetch product name and description from Amazon link"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "TE": "Trailers"
    }
    
    try:
        response = requests.get(amazon_link, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try multiple possible selectors for product name
        product_name = None
        name_selectors = ['#productTitle', '#title', '.product-title']
        for selector in name_selectors:
            element = soup.select_one(selector)
            if element:
                product_name = element.get_text(strip=True)
                break
                
        # Try multiple possible selectors for description
        product_description = None
        desc_selectors = ['#feature-bullets', '#productDescription', '.product-description']
        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                product_description = element.get_text(strip=True)
                break
        
        # If we couldn't find the information, raise an exception
        if not product_name or not product_description:
            print("HTML content:", soup.prettify()[:1000])  # Print first 1000 chars for debugging
            raise Exception("Could not find product information in the page content")
            
        return product_name, product_description
        
    except Exception as e:
        print(f"Error fetching product info: {str(e)}")
        # Fallback to default values if scraping fails
        return "Unknown Product", "No description available"

def generate_loading_tips():
    """Generate content creation tips using OpenAI"""
    prompt = """Generate 10 unique, helpful tips about content creation and video marketing. 
    Format each tip as a single line of text. Make them concise and actionable.
    Focus on social media, video creation, and marketing best practice"""
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    tips = response.choices[0].message.content.strip().split('\n')
    return [tip.strip() for tip in tips if tip.strip()]

# Modify the index route to include AI presenter generation
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Generate loading tips first
            loading_tips = generate_loading_tips()
            
            # Get form data
            amazon_link = request.form.get('amazon_link')
            video_length = request.form.get('video_length', '30')
            platform = request.form.get('platform', 'youtube')
            
            # 1. Get product info from Amazon link
            product_name, product_description = get_product_info(amazon_link)
            if not product_name or not product_description:
                raise Exception("Could not fetch product information from Amazon link.")
            
            # 2. Generate script and visuals separately
            script, visuals = generate_script(product_name, product_description, video_length)
            
            # 3. Fetch images
            images = fetch_images_from_google(product_name)
            
            # 4. Generate voiceover using only the script part
            audio_file = generate_voiceover(script)
            
            # Handle presenter image upload or generate AI presenter
            presenter_image = None
            if 'presenter_image' in request.files:
                file = request.files['presenter_image']
                if file.filename != '':
                    # Save uploaded image
                    presenter_path = "frames/presenter.jpg"
                    os.makedirs('frames', exist_ok=True)
                    file.save(presenter_path)
                    presenter_image = presenter_path
                else:
                    # Generate AI presenter with product context
                    presenter_image = generate_ai_presenter(product_name, product_description)
            else:
                # Generate AI presenter with product context
                presenter_image = generate_ai_presenter(product_name, product_description)
            if not presenter_image:
                raise Exception("Failed to get presenter image")
            
            # 5. Create video with presenter
            video_file = create_video(images, script, presenter_image, product_name, product_description)
            
            if video_file and os.path.exists(video_file):
                print(f"Successfully created video: {video_file}")
                video_filename = os.path.basename(video_file)
                return render_template('result.html', 
                                    main_video=video_filename,  # Just the filename
                                    script=script,
                                    images=images,
                                    loading_tips=[])  # Empty list to avoid undefined error
            else:
                raise Exception("Video creation failed")
                
        except Exception as e:
            print(f"Error in route: {str(e)}")
            traceback.print_exc()
            return render_template('index.html', error=str(e))
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
