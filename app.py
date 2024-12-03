from flask import Flask, render_template, request, send_file
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import numpy as np
from moviepy import ImageSequenceClip, AudioFileClip

app = Flask(__name__)
load_dotenv()

# Initialize API clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

def fetch_images_from_google(query):
    """Fetch images using Google Custom Search API"""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        print("Error: Google API key or CSE ID is not set")
        return []
        
    search_terms = [query, f"{query} product", f"{query} in use"]
    images = []
    
    for term in search_terms:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': term,
            'cx': GOOGLE_CSE_ID,
            'key': GOOGLE_API_KEY,
            'searchType': 'image',
            'num': 3  # Number of images per search term
        }
        
        try:
            response = requests.get(url, params=params)
            print(f"Google API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                results = response.json().get('items', [])
                images.extend([item['link'] for item in results])
            else:
                print(f"Error response from Google: {response.text}")
        except Exception as e:
            print(f"Error fetching images for term '{term}': {str(e)}")
            
    return images[:8]  # Return up to 8 images

def generate_script(product_name, product_description, video_length):
    """Generate video script using OpenAI"""
    prompt = f"""Create a {video_length} second video script to advertise this product:
    Product: {product_name}
    Description: {product_description}
    Include: 
    1. Engaging opening hook
    2. Key features and benefits
    3. Call to action with affiliate link
    Format the response as:
    SCRIPT: (the speaking parts)
    VISUALS: (description of what should be shown)
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def create_video(images, script, audio_file):
    """Create video using MoviePy"""
    try:
        # Create frames directory if it doesn't exist
        os.makedirs('frames', exist_ok=True)
        
        frames = []
        target_size = (1920, 1080)
        
        for idx, img_url in enumerate(images):
            try:
                # Download and save image temporarily
                img_response = requests.get(img_url)
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
                
                # Create black background
                background = Image.new('RGB', target_size, (0, 0, 0))
                
                # Paste image in center
                paste_x = (target_size[0] - new_size[0]) // 2
                paste_y = (target_size[1] - new_size[1]) // 2
                background.paste(img, (paste_x, paste_y))
                
                # Save processed frame
                background.save(temp_img_path, quality=95)
                frames.append(temp_img_path)
                
            except Exception as e:
                print(f"Error processing image {img_url}: {str(e)}")
                continue
        
        if not frames:
            raise Exception("No valid frames were created")
        
        # Create video from frames (5 seconds per image)
        clip = ImageSequenceClip(frames, fps=1/5)
        
        # Create static directory if it doesn't exist
        os.makedirs('static', exist_ok=True)
        
        # Write the result to a file
        output_path = "static/generated_video.mp4"
        clip.write_videofile(
            output_path,
            fps=24,
            codec='libx264'
        )
        
        # Clean up frames
        for frame in frames:
            try:
                os.remove(frame)
            except:
                pass
        try:
            os.rmdir('frames')
        except:
            pass
        
        return output_path
        
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            amazon_link = request.form.get('amazon_link')
            video_length = request.form.get('video_length', '30')
            platform = request.form.get('platform', 'youtube')
            
            # 1. Get product info from Amazon link
            product_name = "Sentro Knitting Machine"
            product_description = "Knitting machine with counter and rotating needles"
            
            # 2. Generate script
            script = generate_script(product_name, product_description, video_length)
            
            # 3. Fetch images
            images = fetch_images_from_google(product_name)
            
            # 4. Generate voiceover
            audio_file = generate_voiceover(script)
            
            # 5. Create video
            video_file = create_video(images, script, audio_file)
            
            return render_template('result.html',
                                 product_name=product_name,
                                 script=script,
                                 images=images,
                                 video_path=video_file)
            
        except Exception as e:
            return render_template('index.html', error=str(e))
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
