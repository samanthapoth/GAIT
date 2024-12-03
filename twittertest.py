from dotenv import load_dotenv
import os
import tweepy

# Load environment variables
load_dotenv()

# Access environment variables
API_KEY = os.getenv('TWITTER_API_KEY')
API_SECRET_KEY = os.getenv('TWITTER_API_SECRET_KEY')
ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

# Create client for v2 API
client = tweepy.Client(
    consumer_key=API_KEY,
    consumer_secret=API_SECRET_KEY,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET
)

try:
    # Get your own user information (this should work with free API)
    me = client.get_me()
    print(f"My username: {me.data.username}")
    print(f"My name: {me.data.name}")
    
except Exception as e:
    print(f"An error occurred: {e}")
