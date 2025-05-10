import lyricsgenius
import json
import time
import os
from tqdm import tqdm

# Genius API credentials
GENIUS_ACCESS_TOKEN = "_YH3-oB4Mcfj2Y0qHI2njg-ROMob53RRhA-lVb0oqiAAWb51IvSeQ75rTZlu5coT"

# Initialize the Genius API client with increased timeout
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=30)

# List of rap artists (for rap songs) - reduced to 5 artists
rap_artists = [
    "Eminem", "Kendrick Lamar", "J. Cole", "Drake", "Jay-Z"
]

# List of non-rap artists (for non-rap songs) - reduced to 5 artists
non_rap_artists = [
    "The Beatles", "Queen", "Elton John", "Adele", "Taylor Swift"
]

def fetch_artist_songs(artist_name, is_rap=True, max_songs=10, max_retries=3):
    """Fetch songs for a given artist with retry logic"""
    for attempt in range(max_retries):
        try:
            artist = genius.search_artist(artist_name, max_songs=max_songs)
            if not artist:
                return []
            
            songs_data = []
            for song in artist.songs:
                # Skip songs without lyrics
                if not song.lyrics:
                    continue
                    
                songs_data.append({
                    "lyrics": song.lyrics,
                    "is_rap": 1 if is_rap else 0,
                    "artist": artist_name,
                    "title": song.title
                })
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
                
            return songs_data
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {artist_name}: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Failed to fetch songs for {artist_name} after {max_retries} attempts")
                return []

def main():
    all_songs = []
    
    # Fetch rap songs
    print("Fetching rap songs...")
    for artist in tqdm(rap_artists):
        songs = fetch_artist_songs(artist, is_rap=True, max_songs=10)
        all_songs.extend(songs)
        print(f"Fetched {len(songs)} songs from {artist}")
    
    # Fetch non-rap songs
    print("\nFetching non-rap songs...")
    for artist in tqdm(non_rap_artists):
        songs = fetch_artist_songs(artist, is_rap=False, max_songs=10)
        all_songs.extend(songs)
        print(f"Fetched {len(songs)} songs from {artist}")
    
    # Save the dataset
    output_file = "lyrics_dataset/genius_dataset.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_songs, f, indent=2)
    
    print(f"\nTotal songs fetched: {len(all_songs)}")
    print(f"Rap songs: {sum(1 for song in all_songs if song['is_rap'] == 1)}")
    print(f"Non-rap songs: {sum(1 for song in all_songs if song['is_rap'] == 0)}")
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    main() 