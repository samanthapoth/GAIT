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

app = Flask(__name__)
load_dotenv()

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
    prompt = f"""Create a {video_length} second video script to advertise this product:
    Product: {product_name}
    Description: {product_description}
    Include: 
    1. Engaging opening hook
    2. Key features and benefits
    3. Call to action with affiliate link
    Format the response EXACTLY like this:
    SCRIPT:
    (The speaking parts here)

    VISUALS:
    (The visual descriptions here)
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    full_response = response.choices[0].message.content
    
    # Split the response into script and visuals
    try:
        script_part = full_response.split('SCRIPT:')[1].split('VISUALS:')[0].strip()
        visuals_part = full_response.split('VISUALS:')[1].strip()
        
        # Store visuals in the session or pass them to the template
        return script_part, visuals_part
    except:
        # Fallback if the splitting fails
        return full_response, ""

def create_talking_avatar(audio_file, presenter_image, script):
    """Create a talking avatar video using D-ID API"""
    try:
        print(f"Starting avatar creation with image: {presenter_image}")
        print(f"Audio file: {audio_file}")
        print(f"Using D-ID API URL: {DID_API_URL}")
        
        # Resize and compress image before upload
        print("Processing image...")
        with Image.open(presenter_image) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate new size while maintaining aspect ratio
            max_size = (800, 800)  # Reasonable size for avatar
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save compressed image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
            img_byte_arr.seek(0)
            
            # Upload the processed image
            print("Attempting to upload presenter image...")
            files = {'image': ('presenter.jpg', img_byte_arr, 'image/jpeg')}
            upload_headers = {
                "Authorization": f"Basic {DID_API_KEY}"
            }
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
                "result_format": "mp4"
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
    """Process presenter image to remove background"""
    try:
        with Image.open(presenter_image) as img:
            # Convert to RGBA for transparency
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Remove background
            print("Removing background from presenter image...")
            img_no_bg = remove(img)
            
            # Save processed image as PNG for transparency
            processed_path = "frames/processed_presenter.png"
            img_no_bg.save(processed_path, quality=95)
            return processed_path
            
    except Exception as e:
        print(f"Error processing presenter image: {str(e)}")
        return presenter_image  # Return original if processing fails

def process_avatar_video(avatar_path):
    """Remove black background from avatar video using unscreen API"""
    try:
        print("Starting unscreen API video processing...")
        
        # Unscreen API endpoint
        UNSCREEN_API_KEY = os.getenv('UNSCREEN_API_KEY')
        unscreen_url = "https://api.unscreen.com/v1/videos"
        
        # Prepare video file for upload
        with open(avatar_path, 'rb') as video_file:
            files = {
                'video': ('avatar.mp4', video_file, 'video/mp4')
            }
            
            headers = {
                'X-Api-Key': UNSCREEN_API_KEY
            }
            
            # Upload video for processing
            print("Uploading video to unscreen...")
            response = requests.post(unscreen_url, headers=headers, files=files)
            
            if response.status_code != 200:
                print(f"Unscreen API error: {response.text}")
                return avatar_path
                
            # Get the job ID
            job_id = response.json()['id']
            print(f"Unscreen job ID: {job_id}")
            
            # Poll for completion
            while True:
                status_response = requests.get(
                    f"{unscreen_url}/{job_id}",
                    headers=headers
                )
                
                status = status_response.json()['status']
                print(f"Processing status: {status}")
                
                if status == 'completed':
                    # Get the processed video URL
                    result_url = status_response.json()['result_url']
                    
                    # Download the processed video
                    print("Downloading processed video...")
                    processed_video = requests.get(result_url)
                    processed_path = "frames/processed_avatar.mp4"
                    
                    with open(processed_path, 'wb') as f:
                        f.write(processed_video.content)
                        
                    print("Processing completed successfully")
                    return processed_path
                    
                elif status == 'failed':
                    print("Unscreen processing failed")
                    return avatar_path
                    
                time.sleep(2)  # Wait before checking again
                
    except Exception as e:
        print(f"Error in unscreen processing: {str(e)}")
        return avatar_path

def create_video(images, script, audio_file, presenter_image=None):
    """Create video using MoviePy with optional talking head"""
    avatar_path = None
    processed_avatar_path = None
    
    try:
        print(f"Starting video creation with presenter_image: {presenter_image}")
        
        # Create frames directory if it doesn't exist
        os.makedirs('frames', exist_ok=True)
        os.makedirs('static', exist_ok=True)
        
        frames = []
        target_size = (1920, 1080)
        
        for idx, img_url in enumerate(images):
            try:
                # Download and save image temporarily
                img_response = requests.get(img_url, timeout=10)
                if img_response.status_code == 200:
                    temp_img_path = f"frames/frame_{idx}.jpg"
                    with open(temp_img_path, "wb") as f:
                        f.write(img_response.content)
                    
                    # Process image with PIL
                    img = Image.open(temp_img_path)
                    
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Calculate resize dimensions
                    aspect_ratio = img.width / img.height
                    if aspect_ratio > target_size[0] / target_size[1]:
                        new_size = (target_size[0], int(target_size[0] / aspect_ratio))
                    else:
                        new_size = (int(target_size[1] * aspect_ratio), target_size[1])
                    
                    # Resize image
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Create transparent background instead of black
                    background = Image.new('RGBA', target_size, (0, 0, 0, 0))
                    
                    # Paste image in center
                    paste_x = (target_size[0] - new_size[0]) // 2
                    paste_y = (target_size[1] - new_size[1]) // 2
                    background.paste(img, (paste_x, paste_y))
                    
                    # Save processed frame as PNG for transparency
                    temp_png_path = f"frames/frame_{idx}.png"
                    background.save(temp_png_path, format='PNG')
                    frames.append(temp_png_path)
                    
            except Exception as e:
                print(f"Error processing image {img_url}: {str(e)}")
                continue
        
        if not frames:
            raise Exception("No valid frames were created")
        
        if presenter_image:
            presenter_image = process_presenter_image(presenter_image)
            # Save directly to static directory
            avatar_path = "static/avatar.mp4"
            avatar_path = create_talking_avatar(audio_file, presenter_image, script)
            
            if avatar_path:
                # Save processed avatar to static directory
                processed_avatar_path = "static/processed_avatar.mp4"
                processed_avatar_path = process_avatar_video(avatar_path)
                
                print(f"Loading processed avatar from: {processed_avatar_path}")
                # Load the processed avatar video
                avatar_clip = VideoFileClip(processed_avatar_path)
                
                # Create main video clip
                main_clip = ImageSequenceClip(frames, durations=[avatar_clip.duration/len(frames)] * len(frames))
                
                # Position avatar in bottom right corner using pos instead of set_position
                avatar_x = main_clip.w - avatar_clip.w - 50
                avatar_y = main_clip.h - avatar_clip.h - 50
                positioned_avatar = avatar_clip.set_pos((avatar_x, avatar_y))  # Changed to set_pos
                
                # Create a black background clip
                background = ColorClip(size=main_clip.size, color=(0,0,0))
                background = background.set_duration(main_clip.duration)
                
                # Composite clips
                final_clip = CompositeVideoClip(
                    [background, main_clip, positioned_avatar],  # Use positioned_avatar
                    size=main_clip.size
                )
                
                # Write final video
                output_path = "static/generated_video.mp4"
                print(f"Writing final video to: {output_path}")
                final_clip.write_videofile(
                    output_path,
                    fps=24,
                    codec='libx264',
                    audio_codec='aac',
                    audio_bitrate="192k"
                )
                
                # Clean up
                avatar_clip.close()
                main_clip.close()
                final_clip.close()
                background.close()
                
                print(f"Video creation completed. Output at: {output_path}")
                return output_path
        
        # If no presenter image or avatar creation failed, create regular video
        clip = ImageSequenceClip(frames, durations=[3] * len(frames))
        clip.write_videofile(
            "static/generated_video.mp4",
            fps=24,
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
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
                                images=images)
            
        except Exception as e:
            return render_template('index.html', error=str(e))
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
