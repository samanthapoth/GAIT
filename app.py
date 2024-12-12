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

app = Flask(__name__)
load_dotenv(override=True)

# Initialize API clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

# Add D-ID API credentials
DID_API_URL = os.getenv('DID_API_URL')
DID_API_KEY = os.getenv('DID_API_KEY')

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
    """Generate video script using OpenAI"""
    prompt = f"""Write a {video_length} second promotional video script for this product:
    Product: {product_name}
    Description: {product_description}
    
    Important: Write ONLY the speaking parts. No labels, no 'SCRIPT:', no 'VISUALS:', no speaker names, no parentheses, no stage directions - just the exact words to be spoken.
    Make it engaging and natural, like someone talking to a friend.
    Include an opening hook, key features, and end with a call to action."""
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    script = response.choices[0].message.content.strip()
    
    # Clean up any remaining formatting
    script = script.replace('SCRIPT:', '').replace('VISUALS:', '')
    script = script.replace('Speaker:', '').replace('Narrator:', '')
    
    # Remove any text within parentheses
    script = re.sub(r'\([^)]*\)', '', script)
    
    # Remove any text within brackets
    script = re.sub(r'\[[^\]]*\]', '', script)
    
    # Remove any lines that look like stage directions
    script = '\n'.join(line for line in script.split('\n') 
                      if not line.strip().startswith(('(', '[', '*', '-')))
    
    return script.strip(), ""  # Return empty string for visuals

def create_talking_avatar(audio_file, presenter_image, script):
    """Create a talking avatar video using D-ID API"""
    try:
        print(f"Starting avatar creation with image: {presenter_image}")
        
        # Get the original MIME type of the uploaded file
        with open(presenter_image, 'rb') as img_file:
            # Upload the original file directly with its original format
            files = {
                'image': (
                    os.path.basename(presenter_image),  # Keep original filename
                    img_file,
                    'image/jpeg' if presenter_image.endswith('.jpg') or presenter_image.endswith('.jpeg') else 'image/png'
                )
            }
            
            upload_headers = {
                "Authorization": f"Basic {DID_API_KEY}"
            }
            
            # Send the original file without any processing
            upload_response = requests.post(
                f"{DID_API_URL}/images",
                headers=upload_headers,
                files=files
            )
        
        print(f"Image upload response status: {upload_response.status_code}")
        print(f"Image upload response: {upload_response.text}")
        
        if upload_response.status_code not in [200, 201]:
            raise Exception(f"Failed to upload image: {upload_response.text}")
            
        # Get the image URL from the response
        image_url = upload_response.json()['url']
        print(f"Got image URL: {image_url}")
        
        # Create the talk
        print("Creating talk with payload...")
        payload = {
            "script": {
                "type": "text",
                "input": script,
                "provider": {
                    "type": "microsoft",
                    "voice_id": "en-US-JennyNeural"
                }
            },
            "config": {
                "fluent": True,
                "pad_audio": 0,
                "result_format": "mp4",
                "remove_background": True,
                "stitch": True
            },
            "source_url": image_url
        }
        print(f"Payload: {payload}")
        
        # Add content-type header for the talk creation
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
        print(f"Talk creation response status: {response.status_code}")
        print(f"Talk creation response: {response.text}")
        
        if response.status_code != 201:
            raise Exception(f"Failed to create talk: {response.text}")
            
        # Get the ID of the created talk
        talk_id = response.json()['id']
        print(f"Got talk ID: {talk_id}")
        
        # Wait for the video to be ready
        print("Waiting for video processing...")
        while True:
            status_response = requests.get(
                f"{DID_API_URL}/talks/{talk_id}",
                headers={"Authorization": f"Basic {DID_API_KEY}"}
            )
            status = status_response.json()['status']
            print(f"Current status: {status}")
            
            if status == 'done':
                result_url = status_response.json()['result_url']
                print(f"Video ready! Result URL: {result_url}")
                
                # Download the video
                print("Downloading video...")
                video_response = requests.get(result_url)
                avatar_path = "frames/avatar.mp4"
                
                with open(avatar_path, 'wb') as f:
                    f.write(video_response.content)
                    
                print(f"Avatar video saved to: {avatar_path}")
                return avatar_path
                
            elif status == 'error':
                error_message = status_response.json().get('error', 'Unknown error')
                print(f"Error in video generation: {error_message}")
                raise Exception(f"Failed to generate avatar video: {error_message}")
                
            time.sleep(2)  # Wait before checking again
            
    except Exception as e:
        print(f"Error creating avatar: {str(e)}")
        print(f"Full error details: {type(e).__name__}: {str(e)}")
        return None

def process_presenter_image(presenter_image):
    """Process presenter image to remove background and compress"""
    try:
        with Image.open(presenter_image) as img:
            # Convert to RGBA for transparency
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Remove background
            print("Removing background from presenter image...")
            img_no_bg = remove(img)
            
            # Calculate new dimensions while maintaining aspect ratio
            max_size = (800, 800)  # Maximum dimensions
            ratio = min(max_size[0] / img_no_bg.size[0], max_size[1] / img_no_bg.size[1])
            new_size = (int(img_no_bg.size[0] * ratio), int(img_no_bg.size[1] * ratio))
            
            # Resize image
            img_no_bg = img_no_bg.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save processed image as compressed JPEG
            processed_path = "frames/processed_presenter.jpg"
            
            # Convert to RGB (removing alpha channel) and save with compression
            rgb_img = Image.new('RGB', img_no_bg.size, (255, 255, 255))
            rgb_img.paste(img_no_bg, mask=img_no_bg.split()[3])  # Use alpha channel as mask
            
            # Save with high compression (lower quality)
            rgb_img.save(processed_path, 'JPEG', quality=85, optimize=True)
            
            # Verify file size
            file_size = os.path.getsize(processed_path) / (1024 * 1024)  # Size in MB
            print(f"Processed image size: {file_size:.2f}MB")
            
            if file_size > 9:  # If still too large, compress more
                for quality in [70, 60, 50, 40]:
                    rgb_img.save(processed_path, 'JPEG', quality=quality, optimize=True)
                    file_size = os.path.getsize(processed_path) / (1024 * 1024)
                    print(f"Recompressed image size at quality {quality}: {file_size:.2f}MB")
                    if file_size < 9:
                        break
            
            return processed_path
            
    except Exception as e:
        print(f"Error processing presenter image: {str(e)}")
        return presenter_image  # Return original if processing fails

def process_avatar_video(avatar_path):
    """Simplified function that just returns the original video path"""
    print("Background removal disabled - using original video")
    return avatar_path

def create_video(images, script, audio_file, presenter_image=None):
    try:
        print(f"Starting video creation with presenter_image: {presenter_image}")
        
        os.makedirs('frames', exist_ok=True)
        os.makedirs('static', exist_ok=True)
        
        # Process frames for main content
        frames = []
        target_size = (1920, 1080)
        
        for idx, img_url in enumerate(images):
            try:
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
                    background = Image.new('RGBA', target_size, (255, 255, 255, 255))
                    
                    # Paste image in center
                    paste_x = (target_size[0] - new_size[0]) // 2
                    paste_y = (target_size[1] - new_size[1]) // 2
                    background.paste(img, (paste_x, paste_y))
                    
                    frames.append(np.array(background))
                    
            except Exception as e:
                print(f"Error processing image {img_url}: {str(e)}")
                continue
        
        if not frames:
            raise Exception("No valid frames were created")
        
        # Load and process avatar video
        if presenter_image:
            avatar_path = create_talking_avatar(audio_file, presenter_image, script)
            if avatar_path and os.path.exists(avatar_path):
                # Create video clips
                main_clip = ImageSequenceClip(frames, durations=[3] * len(frames))
                avatar_clip = VideoFileClip(avatar_path)
                
                # Resize avatar using the resize function directly
                avatar_width = int(target_size[0] * 0.25)  # 25% of screen width
                avatar_clip = resize(avatar_clip, width=avatar_width)
                
                # Calculate new height maintaining aspect ratio
                avatar_height = avatar_clip.h
                
                # Position avatar in bottom right with padding
                avatar_x = target_size[0] - avatar_width - 50
                avatar_y = target_size[1] - avatar_height - 50
                
                # Create the final composite
                final_clip = CompositeVideoClip(
                    [
                        main_clip,
                        avatar_clip.set_position((avatar_x, avatar_y))
                    ],
                    size=target_size
                )
                
                # Write final video
                output_path = "static/generated_video.mp4"
                final_clip.write_videofile(
                    output_path,
                    fps=30,
                    codec='libx264',
                    audio_codec='aac',
                    audio_bitrate="192k"
                )
                
                # Clean up
                main_clip.close()
                avatar_clip.close()
                final_clip.close()
                
                return output_path
                
        # Fallback to basic video if no avatar
        clip = ImageSequenceClip(frames, durations=[3] * len(frames))
        clip.write_videofile(
            "static/generated_video.mp4",
            fps=30,
            codec='libx264',
            audio_codec='aac',
            audio_bitrate="192k"
        )
        
        return "static/generated_video.mp4"
        
    except Exception as e:
        print(f"Error creating video: {str(e)}")
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
    Focus on social media, video creation, and marketing best practices."""
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    tips = response.choices[0].message.content.strip().split('\n')
    return [tip.strip() for tip in tips if tip.strip()]

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
            
            # Handle presenter image upload
            presenter_image = None
            if 'presenter_image' in request.files:
                file = request.files['presenter_image']
                if file.filename != '':
                    # Save uploaded image
                    presenter_path = "frames/presenter.jpg"
                    os.makedirs('frames', exist_ok=True)
                    file.save(presenter_path)
                    presenter_image = presenter_path
            
            # 5. Create video with optional presenter
            video_file = create_video(images, script, audio_file, presenter_image)
            
            # Return the video filename without the 'static/' prefix
            return render_template('result.html', 
                                main_video='generated_video.mp4',  # Remove 'static/' prefix
                                script=script,
                                visuals=visuals,
                                images=images,
                                loading_tips=loading_tips)
            
        except Exception as e:
            return render_template('index.html', error=str(e))
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
