from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

def youtube_authenticate():
    flow = InstalledAppFlow.from_client_secrets_file(
        "client_secret.json", 
        scopes=SCOPES
    )
    credentials = flow.run_local_server()
    return build("youtube", "v3", credentials=credentials)

youtube = youtube_authenticate()
