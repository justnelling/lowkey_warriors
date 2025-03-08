'''
rules: README.md
code: env.py 

we're not considering other file types in the game folders for our embeddings
'''

import os
import json
import openai
import asyncio
import tiktoken
from typing import List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Configuration
TEXTARENA_DIR = "./textarena_games/textarena/envs"
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536  # OpenAI's text-embedding-ada-002 dimension
MAX_TOKENS = 8000  # Leave some buffer below the 8192 limit

def chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> List[str]:
    """Split text into chunks that fit within the token limit"""
    # Initialize tokenizer for ada embedding model
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    
    # Tokenize the text
    tokens = encoding.encode(text)
    
    # If text is already within limits, return it as is
    if len(tokens) <= max_tokens:
        return [text]
    
    # Otherwise, split into chunks
    chunks = []
    current_chunk_tokens = []
    current_token_count = 0
    
    for token in tokens:
        if current_token_count + 1 > max_tokens:
            # Convert tokens back to text and add to chunks
            chunk_text = encoding.decode(current_chunk_tokens)
            chunks.append(chunk_text)
            # Start a new chunk
            current_chunk_tokens = [token]
            current_token_count = 1
        else:
            current_chunk_tokens.append(token)
            current_token_count += 1
    
    # Add the last chunk if it's not empty
    if current_chunk_tokens:
        chunk_text = encoding.decode(current_chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

async def setup_supabase():
    """Set up the Supabase table for game embeddings"""
    # This is a simplified version - you might need to adjust based on Supabase's API
    # You may need to create this table manually in the Supabase dashboard
    
    print("Please make sure you have created the necessary tables in Supabase")
    # Check if table exists, create if it doesn't
    try:
        # Create table with vector extension if needed
        # Note: You'll need to enable the pgvector extension in Supabase
        
        # Example SQL that you might run in Supabase SQL editor:
        """
        -- Enable vector extension
        create extension if not exists vector;

        -- Create table for game embeddings
        create table if not exists game_embeddings (
            id uuid primary key default uuid_generate_v4(),
            game_name text not null,
            content_type text not null, -- 'rules', 'code', 'rules_part1', etc.
            content text not null,
            embedding vector(1536) not null,
            created_at timestamp with time zone default now()
        );

        -- Create HNSW index for faster similarity search
        create index on game_embeddings 
        using hnsw (embedding vector_cosine_ops)
        with (m = 16, ef_construction = 64);
        """
        
        print("Supabase table setup complete")
    except Exception as e:
        print(f"Error setting up Supabase: {e}")
        raise

def generate_embedding(text: str) -> List[float]:
    """Generate embeddings using OpenAI's API"""
    try:
        # Clean and prepare text
        text = text.replace("\n", " ").strip()
        
        # Handle empty text
        if not text:
            return [0.0] * EMBEDDING_DIMENSION
            
        # Generate embedding
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        
        # Convert to list and return
        return list(response.data[0].embedding)
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return [0.0] * EMBEDDING_DIMENSION

def extract_game_info(game_dir: str) -> Dict[str, str]:
    """Extract game information from a game directory"""
    game_name = os.path.basename(game_dir)
    game_info = {
        "name": game_name,
        "rules": "",
        "code": ""
    }
    
    # Extract README content (rules)
    readme_path = os.path.join(game_dir, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            game_info["rules"] = f.read()
    
    # Extract env.py content (code)
    env_path = os.path.join(game_dir, "env.py")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            game_info["code"] = f.read()
    
    return game_info

async def process_game(game_dir: str):
    """Process a single game directory"""
    game_info = extract_game_info(game_dir)
    game_name = game_info["name"]
    
    print(f"Processing game: {game_name}")
    
    # Generate embeddings for rules and code only
    content_types = ["rules", "code"]
    for content_type in content_types:
        content = game_info[content_type]
        
        # Skip if content is empty
        if not content:
            print(f"No {content_type} content found for {game_name}, skipping")
            continue
        
        # Split content into chunks if needed
        chunks = chunk_text(content)
        print(f"{game_name} {content_type} split into {len(chunks)} chunks")
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Generate embedding for this chunk
            try:
                embedding = generate_embedding(chunk)
                
                # Insert into Supabase with chunk number if multiple chunks
                chunk_identifier = f"{content_type}_part{i+1}" if len(chunks) > 1 else content_type
                
                result = supabase.table("game_embeddings").insert({
                    "game_name": game_name,
                    "content_type": chunk_identifier,
                    "content": chunk,
                    "embedding": embedding
                }).execute()
                
                print(f"Stored {game_name} {chunk_identifier} embedding")
                
                # Add a small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing {game_name} {content_type} chunk {i+1}: {str(e)}")
                # Continue with next chunk instead of failing the entire game
                continue

async def main():
    """Main function to process all games"""
    # Setup Supabase
    await setup_supabase()
    
    # Get all game directories
    game_dirs = []
    for item in os.listdir(TEXTARENA_DIR):
        item_path = os.path.join(TEXTARENA_DIR, item)
        if os.path.isdir(item_path):
            game_dirs.append(item_path)
    
    print(f"Found {len(game_dirs)} games")
    
    # Process each game
    for game_dir in game_dirs:
        try:
            await process_game(game_dir)
        except Exception as e:
            print(f"Error processing game directory {game_dir}: {str(e)}")
            # Continue with next game instead of failing the entire process
            continue
    
    print("All games processed and embedded")

if __name__ == "__main__":
    asyncio.run(main())