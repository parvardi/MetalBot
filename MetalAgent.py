
import os
import streamlit as st
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.tools import tool
from langchain.schema import SystemMessage as SchemaSystemMessage
from langchain.schema import HumanMessage as SchemaHumanMessage
from langchain.schema import AIMessage as SchemaAIMessage

import random
import glob
import re
from datetime import datetime, timedelta

from embed_helper import load_album_data, search_album_id, generate_spotify_embed
# Load album dataset
ALBUM_DATA_PATH = 'spotify_album_ids.csv'  # Replace with your path
album_data = load_album_data(ALBUM_DATA_PATH)

# Initialize the LLM with your API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY environment variable not set.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Use the desired model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Define RequestState
class RequestState(TypedDict):
    """State representing the metalhead's request conversation."""
    messages: list
    request: list[str]
    finished: bool

prompt = '''
    You are MetalAgent, an interactive metal recommendation system. 
    **IMPORTANT: Whenever you provide recommendations, output them in a bullet list in the following format, making sure to include release dates:**
    - Artist - Release [Genre] (Release Date)

    **Example:**
    - Metallica - Master of Puppets [Thrash Metal] (March 3, 1986)
    - Iron Maiden - The Number of the Beast [Heavy Metal] (March 22, 1982)

    "Use the provided tools to assist you:\n"
    "- **get_menu**: Retrieves the latest genre menu and recent releases in a specified genre.\n"
    "- **recommend_releases**: Provide the most recent releases in the specified genres from the past month.\n\n"
'''


# The system instruction as a SystemMessage
METALAGENT_SYSINT = SystemMessage(content=prompt)

WELCOME_MSG = '''
Welcome to the MetalAgent recommendation system! Type q to quit. What metal genre do you have in mind today?\n 
Example: give me a list of recent atmospheric black metal releases with release dates.
'''

# Adjust the paths to match your local directory structure
Subgenres_path = 'data/subgenres.txt'

# Get all album files from the 'data' directory
album_files = glob.glob('data/*.txt')
album_files = [f for f in album_files if 'subgenres.txt' not in f]

# Load subgenres
try:
    with open(Subgenres_path, 'r', encoding='utf-8') as file:
        subgenres_text = file.read()
except FileNotFoundError:
    st.error(f"Subgenres file not found at {Subgenres_path}")
    st.stop()

# Load and parse album releases
all_releases = []

for path in album_files:
    try:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Split releases by separator
            releases = content.strip().split('----------------------------------------')
            for release in releases:
                if release.strip():
                    # Parse the release details into a dict
                    release_data = {}
                    for line in release.strip().split('\n'):
                        if ': ' in line:
                            key, value = line.split(': ', 1)
                            release_data[key.strip()] = value.strip()
                    all_releases.append(release_data)
    except FileNotFoundError:
        st.error(f"Release file not found at {path}")
        st.stop()

# Function to parse release date strings
def parse_release_date(date_str):
    """Parse a release date string into a datetime object."""
    # Remove ordinal suffixes (st, nd, rd, th)
    date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
    try:
        date = datetime.strptime(date_str.strip(), '%B %d, %Y')
    except ValueError:
        date = None
    return date


# Updated get_recent_releases function
def get_recent_releases(genre: str = None) -> str:
    """Compile recent releases into a formatted string, filtered by genre if provided."""
    if genre:
        # Filter releases by genre
        filtered_releases = [release for release in all_releases if genre.lower() in release.get('Genre', '').lower()]
    else:
        filtered_releases = all_releases
    # Limit to the most recent 100 releases
    sample_size = min(100, len(filtered_releases))
    recent_releases = filtered_releases[-sample_size:]  # Assuming releases are in chronological order
    # Format releases as a bulleted list
    releases_text = '\n'.join([
        f"- {release.get('Band', 'N/A')} - {release.get('Album', 'N/A')} [{release.get('Genre', 'N/A')}] ({release.get('Release Date', 'N/A')})"
        for release in recent_releases
    ])
    return releases_text

# Define tools

@tool
def get_menu(genre: str) -> str:
    """Provide the latest up-to-date genre menu and recent releases in the specified genre."""
    recent_releases = get_recent_releases(genre)
    return f"""
GENRE MENU:
{subgenres_text}

Recent Releases in {genre} (Limited to 100 entries to fit model constraints):
{recent_releases}
"""

@tool
def recommend_releases(genres: str) -> str:
    """Provide the most recent releases in the specified genres from the past month."""
    genres_list = [genre.strip() for genre in genres.split(',')]
    matching_releases = []
    current_date = datetime.now()
    one_month_ago = current_date - timedelta(days=30)
    for release in all_releases:
        release_genre = release.get('Genre', '').lower()
        if any(genre.lower() in release_genre for genre in genres_list):
            release_date_str = release.get('Release Date', '')
            release_date = parse_release_date(release_date_str)
            if release_date and one_month_ago <= release_date <= current_date:
                matching_releases.append(release)
    if not matching_releases:
        return "No recent releases found for the specified genres."
    # Limit to top 15 releases
    matching_releases = matching_releases[:15]
    # Format releases as a bulleted list
    releases_text = '\n'.join([
        f"- {release.get('Band', 'N/A')} - {release.get('Album', 'N/A')} [{release.get('Genre', 'N/A')}] ({release.get('Release Date', 'N/A')})"
        for release in matching_releases
    ])
    return releases_text

# Define the tools
tools = [get_menu, recommend_releases]

# Initialize the agent with tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True
)

def parse_recommendations(result):
    """Extract album and artist information from recommendations."""
    recommendations = []
    lines = result.strip().split('\n')
    for line in lines:
        # Remove leading dash and whitespace
        line = line.strip()
        if line.startswith('- '):
            line = line[2:]
            # Split the line on ' - '
            parts = line.split(' - ', 1)
            if len(parts) >= 2:
                artist = parts[0].strip()
                album_part = parts[1].strip()
                # Remove genre and release date if present
                album = re.sub(r'\s*\[.*?\]', '', album_part)
                album = re.sub(r'\s*\(.*?\)', '', album)
                recommendations.append((artist, album.strip()))
            else:
                # Debugging: Log the unmatched line
                st.write(f"Could not parse line: {line}")
    return recommendations


def main():
    st.title("MetalAgent Recommendation System")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.finished = False

    # Render messages from session state
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                if "iframe" in msg:
                    st.components.v1.html(msg["iframe"], height=400)
                elif "content" in msg:
                    st.write(msg["content"])

    if st.session_state.finished:
        st.write("**Conversation ended.**")
        return

    user_input = st.chat_input("Your response:")
    if user_input:
        if user_input.lower() in {"q", "quit", "exit", "goodbye"}:
            st.session_state.finished = True
            st.write("**Conversation ended.**")
        else:
            # Build chat history excluding iframe messages
            chat_history = []
            for msg in st.session_state.messages:
                if "content" in msg:
                    if msg["role"] == "user":
                        chat_history.append(SchemaHumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(SchemaAIMessage(content=msg["content"]))

            # Get assistant response
            assistant_response = agent.run(input=user_input, chat_history=chat_history)
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

            # Parse recommendations and append iframe messages
            recommendations = parse_recommendations(assistant_response)
            print(recommendations)
            if recommendations:
                for artist, album in recommendations:
                    album_id = search_album_id(album_data, artist, album)
                    if album_id:
                        iframe = f"""                               
                        <iframe src="https://open.spotify.com/embed/album/{album_id}" 
                                width="100%" height="380" frameborder="0" 
                                allowtransparency="true" allow="encrypted-media"></iframe>
                        """
                        st.session_state.messages.append({"role": "assistant", "iframe": iframe})

            st.rerun()

    elif not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": WELCOME_MSG})
        st.rerun()
    else:
        st.stop()


if __name__ == "__main__":
    main()