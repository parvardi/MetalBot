import pandas as pd

def load_album_data(file_path):
    """
    Load album data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing album data.

    Returns:
        pd.DataFrame: DataFrame containing album and artist information with Spotify Album IDs.
    """
    return pd.read_csv(file_path)

def search_album_id(album_data, artist_name, album_name):
    """
    Search for the Spotify Album ID using artist and album names.

    Args:
        album_data (pd.DataFrame): DataFrame containing album and artist data.
        artist_name (str): Name of the artist (case-insensitive).
        album_name (str): Name of the album (case-insensitive).

    Returns:
        str or None: The Spotify Album ID if found, otherwise None.
    """
    # Normalize input to lowercase for case-insensitive comparison
    artist_name = artist_name.lower()
    album_name = album_name.lower()

    # Search for the matching album and artist
    match = album_data[(album_data['artist_lower'] == artist_name) & 
                       (album_data['album_lower'] == album_name)]

    if not match.empty:
        return match.iloc[0]['Spotify Album ID']
    else:
        return None

def generate_spotify_embed(album_id):
    """
    Generates an embedded Spotify album iframe given an album ID.

    Args:
        album_id (str): The Spotify album ID.

    Returns:
        str: HTML iframe string for embedding the Spotify album.
    """
    # Base URL for Spotify embed
    base_url = "https://open.spotify.com/embed/album/"

    # Construct the full URL for the iframe
    embed_url = f"{base_url}{album_id}"

    # Generate the iframe HTML
    iframe_html = (
        f'<iframe src="{embed_url}" width="300" height="380" frameborder="0" '
        f'allowtransparency="true" allow="encrypted-media"></iframe>'
    )

    return iframe_html

# Example usage
if __name__ == "__main__":
    # Load album data
    file_path = input("Enter the path to the album CSV file: ")
    album_data = load_album_data(file_path)

    # Input artist and album names
    artist_name = input("Enter artist name: ")
    album_name = input("Enter album name: ")

    # Search for the album ID
    album_id = search_album_id(album_data, artist_name, album_name)

    if album_id:
        # Generate embed code
        embed_code = generate_spotify_embed(album_id)
        print("Generated Embed Code:")
        print(embed_code)
    else:
        print("Album or artist not found in the data.")
