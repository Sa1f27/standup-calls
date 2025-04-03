from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pymongo import MongoClient
import gridfs
import time
import os
import ffmpeg
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import re
from datetime import datetime

# Google Drive Authentication
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

# Connect to Google Drive API for efficient downloading
credentials = gauth.credentials
drive_service = build('drive', 'v3', credentials=credentials)

# MongoDB Connection
client = MongoClient("mongodb://192.168.48.112:27017/")  # Update with your MongoDB URI
db = client["video_db"]
fs = gridfs.GridFS(db)
collection = db["batch_04"]

def fetch_videos_from_drive():
    """Fetch all MP4 videos from the 'Meet Recordings' folder in Google Drive."""
    folder_name = 'Meet Recordings'
    folder_list = drive.ListFile({'q': f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder'"}).GetList()
    
    if not folder_list:
        print(f"❌ Folder '{folder_name}' not found.")
        return []

    folder_id = folder_list[0]['id']
    
    query = f"'{folder_id}' in parents and mimeType='video/mp4'"
    file_list = drive.ListFile({'q': query}).GetList()

    return [{"id": file['id'], "name": file['title']} for file in file_list]

def sanitize_filename(filename):
    """Sanitize filenames to avoid issues with invalid characters."""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)  # Replace invalid characters
    filename = filename.replace(":", "_")  # Replace colons (specific to Windows)
    filename = filename.replace(" ", "_")  # Replace spaces with underscores
    return filename

def get_new_videos():
    """Get new videos that are not already stored in MongoDB."""
    existing_ids = {video["file_id"] for video in collection.find({}, {"file_id": 1})}
    all_videos = fetch_videos_from_drive()
    return [video for video in all_videos if video["id"] not in existing_ids]

def download_video(file_id, file_name):
    """Download video from Google Drive in chunks to prevent memory issues."""
    request = drive_service.files().get_media(fileId=file_id)
    sanitized_file_name = sanitize_filename(file_name)  # Sanitize the filename
    local_path = f"temp_{sanitized_file_name}"

    with open(local_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"⬇️ Downloading {file_name}: {int(status.progress() * 100)}%")

    print(f"✅ Download complete: {local_path}")
    return local_path

def compress_video(input_file):
    """Compress video using FFmpeg with optimal settings for smaller size and high quality."""
    sanitized_filename = sanitize_filename(os.path.basename(input_file))  # Sanitize the filename to avoid illegal characters.
    
    # Explicitly set the output file extension to .mp4
    output_file = f"compressed_{sanitized_filename}.mp4"  # Ensure proper extension (.mp4)

    try:
        # Run the compression with ffmpeg
        (
            ffmpeg
            .input(input_file)
            .output(output_file, vcodec="libx264", preset="slow", crf=23)
            .run(overwrite_output=True)
        )
        print(f"✅ Compression complete: {output_file}")
        return output_file
    except ffmpeg._run.Error as e:
        print(f"❌ Error during compression: {e.stderr.decode()}")
        return None

def extract_audio(video_file):
    """Extract audio from video for speech-to-text processing."""
    audio_file = video_file.replace(".mp4", ".wav")
    (
        ffmpeg
        .input(video_file)
        .output(audio_file, acodec='pcm_s16le', ar='16000')
        .run(overwrite_output=True)
    )
    print(f"🎙️ Audio extraction complete: {audio_file}")
    return audio_file

def upload_to_mongodb(file_name, file_id, video_path):
    """Upload video file to MongoDB GridFS along with date and time."""
    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(video_path, "rb") as f:
        video_id = fs.put(f, filename=file_name)

    collection.insert_one({
        "file_name": file_name,
        "file_id": file_id,
        "video_id": video_id,
        "upload_datetime": current_datetime  # Add the current date and time
    })
    print(f"✅ Uploaded {file_name} to MongoDB (ID: {video_id})")

    os.remove(video_path)

def process_videos():
    """Process and upload new videos to MongoDB."""
    print("🔍 Checking for new videos...")
    new_videos = get_new_videos()

    if not new_videos:
        print("✅ No new videos found.")
        return

    print(f"📥 Found {len(new_videos)} new videos.")
    for video in new_videos:
        print(f"⬇️ Downloading: {video['name']} (ID: {video['id']})")
        original_file = download_video(video['id'], video['name'])

        print(f"⚡ Compressing: {original_file}")
        compressed_file = compress_video(original_file)

        if compressed_file:
            print(f"⬆️ Uploading compressed file to MongoDB...")
            upload_to_mongodb(video['name'], video['id'], compressed_file)

            os.remove(original_file)

    print("✅ All new videos uploaded to MongoDB.")

if __name__ == "__main__":
    while True:
        print("\n🔄 Scanning Google Drive for new videos...")
        process_videos()
        print("⏳ Waiting for 60 seconds before next scan...")
        time.sleep(60)
