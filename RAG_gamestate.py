import os
import json
import asyncio
import openai
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime
import uuid
import numpy as np

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536  # OpenAI's text-embedding-ada-002 dimension

class GameStateManager:
    """Manages game state storage and retrieval for the TextArena agent"""
    
    def __init__(self):
        """Initialize the GameStateManager"""
        self.current_game_id = None
        self.current_game_name = None
        self.current_turn = 0
        self.setup_database()
    
    def setup_database(self):
        """Set up the Supabase tables for game state storage"""
        print("Please make sure you have created the necessary tables in Supabase")
        
        # SQL to run in Supabase SQL editor:
        """
        -- Enable vector extension if not already enabled
        CREATE EXTENSION IF NOT EXISTS vector;

        -- Game sessions table - stores metadata about each game session
        CREATE TABLE IF NOT EXISTS game_sessions (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            game_name TEXT NOT NULL,
            start_time TIMESTAMP WITH TIME ZONE DEFAULT now(),
            end_time TIMESTAMP WITH TIME ZONE,
            outcome TEXT, -- 'win', 'loss', 'draw'
            outcome_reason TEXT, -- Reason for the outcome from info['reason']
            score FLOAT,
            opponent TEXT,
            total_turns INTEGER DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );

        -- Game turns table - stores individual turns with observations and actions
        CREATE TABLE IF NOT EXISTS game_turns (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            session_id UUID REFERENCES game_sessions(id),
            turn_number INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            observation TEXT NOT NULL,
            observation_embedding VECTOR(1536),
            action TEXT,
            reward FLOAT DEFAULT 0.0,
            reasoning TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );

        -- Game patterns table - stores recognized patterns in gameplay
        CREATE TABLE IF NOT EXISTS game_patterns (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            game_name TEXT NOT NULL,
            pattern_type TEXT NOT NULL, -- 'opening', 'response', 'strategy', 'mistake'
            observation_pattern TEXT NOT NULL,
            recommended_action TEXT NOT NULL,
            success_rate FLOAT DEFAULT 0.0,
            usage_count INTEGER DEFAULT 0,
            pattern_embedding VECTOR(1536),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );

        -- Game outcomes table - stores specific outcomes and their causes
        CREATE TABLE IF NOT EXISTS game_outcomes (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            game_name TEXT NOT NULL,
            outcome TEXT NOT NULL, -- 'win', 'loss', 'draw'
            reason TEXT NOT NULL,
            frequency INTEGER DEFAULT 1,
            last_seen TIMESTAMP WITH TIME ZONE DEFAULT now(),
            reason_embedding VECTOR(1536),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );

        -- Create HNSW indexes for faster similarity search
        CREATE INDEX IF NOT EXISTS game_turns_embedding_idx ON game_turns 
        USING hnsw (observation_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);

        CREATE INDEX IF NOT EXISTS game_patterns_embedding_idx ON game_patterns 
        USING hnsw (pattern_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        
        CREATE INDEX IF NOT EXISTS game_outcomes_embedding_idx ON game_outcomes 
        USING hnsw (reason_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
        
        print("Game state database setup complete")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding
        """
        # Limit text size to avoid token limits
        if len(text) > 8000:
            text = text[:8000]
            
        try:
            # Try OpenAI first
            if openai_client:
                response = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                embedding = response.data[0].embedding
                # Ensure it's a list of floats
                return [float(x) for x in embedding]
            
            # Fall back to Anthropic
            elif self.anthropic_client:
                response = self.anthropic_client.embeddings.create(
                    model="claude-3-haiku-20240307",
                    input=text
                )
                embedding = response.embeddings[0]
                # Ensure it's a list of floats
                return [float(x) for x in embedding]
            
            else:
                print("No embedding client available")
                # Return a zero vector as fallback
                return [0.0] * EMBEDDING_DIMENSION
                
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * EMBEDDING_DIMENSION
    
    async def start_game_session(self, game_name: str, opponent: str = "unknown") -> str:
        """
        Start a new game session and return the session ID
        
        Args:
            game_name: Name of the game being played
            opponent: Identifier for the opponent
            
        Returns:
            session_id: UUID of the created game session
        """
        try:
            result = supabase.table("game_sessions").insert({
                "game_name": game_name,
                "opponent": opponent,
                "total_turns": 0
            }).execute()
            
            session_id = result.data[0]["id"]
            self.current_game_id = session_id
            self.current_game_name = game_name
            self.current_turn = 0
            
            print(f"Started new game session: {session_id} for game: {game_name}")
            return session_id
        except Exception as e:
            print(f"Error starting game session: {str(e)}")
            # Generate a local ID as fallback
            fallback_id = str(uuid.uuid4())
            self.current_game_id = fallback_id
            self.current_game_name = game_name
            self.current_turn = 0
            return fallback_id
    
    async def end_game_session(self, session_id: str, outcome: str, info: Dict[str, Any], score: float = 0.0):
        """
        End a game session with the final outcome and info
        
        Args:
            session_id: UUID of the game session
            outcome: Result of the game ('win', 'loss', 'draw')
            info: Info dictionary from the environment, containing 'reason'
            score: Numerical score or reward from the game
        """
        try:
            # Extract reason from info dictionary
            reason = info.get('reason', 'No reason provided')
            
            # Update the game session
            supabase.table("game_sessions").update({
                "end_time": datetime.now().isoformat(),
                "outcome": outcome,
                "outcome_reason": reason,
                "score": score
            }).eq("id", session_id).execute()
            
            print(f"Ended game session: {session_id} with outcome: {outcome}, reason: {reason}")
            
            # Store the outcome reason for future reference
            await self.store_game_outcome(self.current_game_name, outcome, reason)
            
            # Generate patterns from this completed game
            await self.generate_game_patterns(session_id)
        except Exception as e:
            print(f"Error ending game session: {str(e)}")
    
    async def store_turn(self, 
                        session_id: str, 
                        player_id: int, 
                        observation: str, 
                        action: Optional[str] = None, 
                        reward: float = 0.0,
                        reasoning: str = ""):
        """
        Store a game turn with the agent's observation and action
        
        Args:
            session_id: UUID of the game session
            player_id: ID of the player (usually 0 for the agent)
            observation: Text observation from the environment
            action: Action taken by the agent (can be None if storing before action)
            reward: Reward received for the action
            reasoning: Agent's reasoning for taking the action
        """
        try:
            # Increment turn counter
            self.current_turn += 1
            
            # Generate embedding for the observation
            embedding = self.generate_embedding(observation)
            
            # Store the game turn
            supabase.table("game_turns").insert({
                "session_id": session_id,
                "turn_number": self.current_turn,
                "player_id": player_id,
                "observation": observation,
                "observation_embedding": embedding,
                "action": action,
                "reward": reward,
                "reasoning": reasoning
            }).execute()
            
            # Update total turns in the session
            supabase.table("game_sessions").update({
                "total_turns": self.current_turn
            }).eq("id", session_id).execute()
            
            print(f"Stored game turn for session: {session_id}, turn: {self.current_turn}")
        except Exception as e:
            print(f"Error storing game turn: {str(e)}")
    
    async def update_turn_action(self, session_id: str, turn_number: int, action: str, reasoning: str = ""):
        """
        Update a previously stored turn with the action taken
        
        Args:
            session_id: UUID of the game session
            turn_number: Turn number to update
            action: Action taken by the agent
            reasoning: Agent's reasoning for taking the action
        """
        try:
            supabase.table("game_turns").update({
                "action": action,
                "reasoning": reasoning
            }).eq("session_id", session_id).eq("turn_number", turn_number).execute()
            
            print(f"Updated turn action for session: {session_id}, turn: {turn_number}")
        except Exception as e:
            print(f"Error updating turn action: {str(e)}")
    
    async def retrieve_similar_observations(self, observation: str, game_name: str = None, limit: int = 5) -> List[Dict]:
        """
        Retrieve similar observations and their successful actions
        
        Args:
            observation: Current game observation
            game_name: Optional filter for specific game
            limit: Maximum number of observations to retrieve
            
        Returns:
            List of similar observations with their actions and reasoning
        """
        try:
            # Generate embedding for the observation
            embedding = self.generate_embedding(observation)
            
            # Use direct query instead of exec_sql function
            # First, get all game turns with embeddings
            if game_name:
                # Filter by game name
                result = supabase.table("game_turns").select("*, game_sessions(game_name, outcome, outcome_reason)").execute()
                
                # Filter results manually
                filtered_data = []
                for item in result.data:
                    if (item.get("game_sessions", {}).get("game_name") == game_name and
                        item.get("observation_embedding") is not None and
                        item.get("action") is not None):
                        filtered_data.append(item)
                
                result.data = filtered_data
            else:
                # Get all games
                result = supabase.table("game_turns").select("*, game_sessions(game_name, outcome, outcome_reason)").execute()
                
                # Filter results manually
                filtered_data = []
                for item in result.data:
                    if (item.get("observation_embedding") is not None and
                        item.get("action") is not None):
                        filtered_data.append(item)
                
                result.data = filtered_data
            
            # If we have results, we'll need to sort them manually by similarity
            if result.data:
                # Calculate similarity for each result
                for item in result.data:
                    # Skip items without embeddings
                    if not item.get("observation_embedding"):
                        item["similarity"] = 0
                        continue
                    
                    # Calculate cosine similarity
                    item_embedding = item["observation_embedding"]
                    similarity = self._calculate_similarity(embedding, item_embedding)
                    item["similarity"] = similarity
                
                # Sort by similarity (highest first)
                result.data.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                
                # Limit to the requested number
                result.data = result.data[:limit]
            
            return result.data
        except Exception as e:
            print(f"Error retrieving similar observations: {str(e)}")
            return []
    
    def _calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert embeddings to proper format
            # Case 1: Handle string embeddings
            if isinstance(embedding1, str):
                try:
                    # Try to parse as JSON
                    if embedding1.startswith('[') and embedding1.endswith(']'):
                        embedding1 = json.loads(embedding1)
                    else:
                        # If it's not JSON, try to convert directly to float list
                        # First, clean the string by removing brackets and splitting by commas
                        cleaned = embedding1.strip('[]').replace(' ', '').split(',')
                        embedding1 = [float(x) for x in cleaned if x]
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing embedding1: {e}")
                    # Create a zero vector as fallback
                    embedding1 = [0.0] * EMBEDDING_DIMENSION
            
            if isinstance(embedding2, str):
                try:
                    # Try to parse as JSON
                    if embedding2.startswith('[') and embedding2.endswith(']'):
                        embedding2 = json.loads(embedding2)
                    else:
                        # If it's not JSON, try to convert directly to float list
                        # First, clean the string by removing brackets and splitting by commas
                        cleaned = embedding2.strip('[]').replace(' ', '').split(',')
                        embedding2 = [float(x) for x in cleaned if x]
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing embedding2: {e}")
                    # Create a zero vector as fallback
                    embedding2 = [0.0] * EMBEDDING_DIMENSION
            
            # Case 2: Handle PostgreSQL vector type
            # PostgreSQL vector might be represented as a list-like object but not a Python list
            if hasattr(embedding1, '__iter__') and not isinstance(embedding1, (list, np.ndarray)):
                embedding1 = list(embedding1)
            
            if hasattr(embedding2, '__iter__') and not isinstance(embedding2, (list, np.ndarray)):
                embedding2 = list(embedding2)
            
            # Convert to numpy arrays with explicit float type
            try:
                vec1 = np.array(embedding1, dtype=np.float64)
                vec2 = np.array(embedding2, dtype=np.float64)
                
                # Ensure vectors have the same dimension
                if len(vec1) != len(vec2):
                    print(f"Warning: Embedding dimensions don't match: {len(vec1)} vs {len(vec2)}")
                    # Pad the shorter one with zeros
                    if len(vec1) < len(vec2):
                        vec1 = np.pad(vec1, (0, len(vec2) - len(vec1)), 'constant')
                    else:
                        vec2 = np.pad(vec2, (0, len(vec1) - len(vec2)), 'constant')
                
                # Calculate cosine similarity
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0
                    
                similarity = dot_product / (norm1 * norm2)
                return similarity
                
            except (ValueError, TypeError) as e:
                print(f"Error converting embeddings to numeric arrays: {e}")
                return 0
                
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0
    
    async def retrieve_game_patterns(self, game_name: str, observation: str, limit: int = 3) -> List[Dict]:
        """
        Retrieve relevant game patterns based on the current observation
        
        Args:
            game_name: Name of the game to get patterns for
            observation: Current observation to match against patterns
            limit: Maximum number of patterns to retrieve
            
        Returns:
            List of relevant game patterns with recommended actions
        """
        try:
            # Generate embedding for the observation
            embedding = self.generate_embedding(observation)
            
            # Limit observation size to avoid 414 errors
            short_observation = observation[:500] + "..." if len(observation) > 500 else observation
            
            # Get all patterns for this game
            result = supabase.table("game_patterns").select("*").eq("game_name", game_name).execute()
            
            # Filter results manually
            filtered_data = []
            for item in result.data:
                if item.get("pattern_embedding") is not None:
                    filtered_data.append(item)
            
            result.data = filtered_data
            
            # If we have results, we'll need to sort them manually by similarity
            if result.data:
                # Calculate similarity for each result
                for item in result.data:
                    # Skip items without embeddings
                    if not item.get("pattern_embedding"):
                        item["similarity"] = 0
                        continue
                    
                    # Calculate cosine similarity
                    item_embedding = item["pattern_embedding"]
                    similarity = self._calculate_similarity(embedding, item_embedding)
                    item["similarity"] = similarity
                
                # Sort by similarity (highest first)
                result.data.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                
                # Limit to the requested number
                result.data = result.data[:limit]
            
            return result.data
        except Exception as e:
            print(f"Error retrieving game patterns: {str(e)}")
            return []
    
    async def store_game_outcome(self, game_name: str, outcome: str, reason: str):
        """
        Store or update a game outcome and its reason
        
        Args:
            game_name: Name of the game
            outcome: Result of the game ('win', 'loss', 'draw')
            reason: Reason for the outcome from info['reason']
        """
        try:
            # Generate embedding for the reason
            embedding = self.generate_embedding(reason)
            
            # Check if this outcome reason already exists
            result = supabase.table("game_outcomes").select("*").eq("game_name", game_name).eq("outcome", outcome).eq("reason", reason).execute()
            
            if result.data and len(result.data) > 0:
                # Update existing outcome
                outcome_id = result.data[0]["id"]
                frequency = result.data[0]["frequency"] + 1
                
                supabase.table("game_outcomes").update({
                    "frequency": frequency,
                    "last_seen": datetime.now().isoformat()
                }).eq("id", outcome_id).execute()
                
                print(f"Updated existing outcome for {game_name}: {outcome}, frequency: {frequency}")
            else:
                # Create new outcome
                supabase.table("game_outcomes").insert({
                    "game_name": game_name,
                    "outcome": outcome,
                    "reason": reason,
                    "reason_embedding": embedding
                }).execute()
                
                print(f"Stored new outcome for {game_name}: {outcome}")
        except Exception as e:
            print(f"Error storing game outcome: {str(e)}")
    
    async def generate_game_patterns(self, session_id: str):
        """
        Generate patterns from a completed game session
        
        Args:
            session_id: UUID of the completed game session
        """
        try:
            # Get the game session details
            session_result = supabase.table("game_sessions").select("*").eq("id", session_id).execute()
            
            if not session_result.data:
                print(f"No game session found with ID: {session_id}")
                return
            
            session = session_result.data[0]
            game_name = session["game_name"]
            outcome = session["outcome"]
            
            # Only generate patterns for winning games
            if outcome != "win":
                print(f"Not generating patterns for non-winning game: {session_id}")
                return
            
            # Get all turns for this session
            turns_result = supabase.table("game_turns").select("*").eq("session_id", session_id).order("turn_number").execute()
            
            if not turns_result.data:
                print(f"No game turns found for session: {session_id}")
                return
            
            turns = turns_result.data
            
            # Prepare data for pattern generation
            game_summary = {
                "game_name": game_name,
                "outcome": outcome,
                "reason": session["outcome_reason"],
                "score": session["score"],
                "turns": len(turns),
                "turn_data": [
                    {
                        "turn": turn["turn_number"],
                        "observation": turn["observation"],
                        "action": turn["action"],
                        "reasoning": turn["reasoning"]
                    }
                    for turn in turns
                ]
            }
            
            # Generate patterns using LLM
            prompt = f"""
            Analyze this winning game session and extract 3 key patterns that led to success:
            
            Game: {game_name}
            Outcome: {outcome}
            Reason: {session['outcome_reason']}
            
            Game Play:
            {json.dumps(game_summary['turn_data'], indent=2)}
            
            For each pattern, provide:
            1. Pattern Type (opening, response, strategy, mistake)
            2. Observation Pattern (a generalized description of the game state)
            3. Recommended Action (what action to take when this pattern is observed)
            4. Success Rate Estimate (0.0 to 1.0)
            
            Format your response as a JSON array of objects with these fields.
            """
            
            # Use OpenAI to generate patterns
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a game strategy analyst. Extract patterns from successful game sessions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Parse patterns from the response
            try:
                patterns_text = response.choices[0].message.content
                # Extract JSON array from the response
                json_match = re.search(r'\[.*\]', patterns_text, re.DOTALL)
                if json_match:
                    patterns = json.loads(json_match.group(0))
                else:
                    print(f"No JSON patterns found in response: {patterns_text}")
                    return
                
                # Store each pattern
                for pattern in patterns:
                    observation_pattern = pattern.get("observation_pattern", "")
                    recommended_action = pattern.get("recommended_action", "")
                    
                    if not observation_pattern or not recommended_action:
                        continue
                    
                    # Generate embedding for the observation pattern
                    embedding = self.generate_embedding(observation_pattern)
                    
                    # Check if a similar pattern already exists
                    similar_patterns = supabase.table("game_patterns").select("*").eq("game_name", game_name).order(f"pattern_embedding <=> '{embedding}'::vector").limit(1).execute()
                    
                    if similar_patterns.data and len(similar_patterns.data) > 0:
                        # Update existing pattern
                        pattern_id = similar_patterns.data[0]["id"]
                        usage_count = similar_patterns.data[0]["usage_count"] + 1
                        current_success_rate = similar_patterns.data[0]["success_rate"]
                        
                        # Update success rate as a weighted average
                        new_success_rate = ((current_success_rate * (usage_count - 1)) + pattern.get("success_rate", 0.5)) / usage_count
                        
                        supabase.table("game_patterns").update({
                            "usage_count": usage_count,
                            "success_rate": new_success_rate,
                            "updated_at": datetime.now().isoformat()
                        }).eq("id", pattern_id).execute()
                        
                        print(f"Updated existing pattern for {game_name}, usage count: {usage_count}")
                    else:
                        # Store the new pattern
                        supabase.table("game_patterns").insert({
                            "game_name": game_name,
                            "pattern_type": pattern.get("pattern_type", "strategy"),
                            "observation_pattern": observation_pattern,
                            "recommended_action": recommended_action,
                            "success_rate": pattern.get("success_rate", 0.5),
                            "usage_count": 1,
                            "pattern_embedding": embedding
                        }).execute()
                        
                        print(f"Stored new pattern for {game_name}")
                
                print(f"Generated and stored {len(patterns)} patterns for game: {game_name}")
            except Exception as e:
                print(f"Error parsing or storing patterns: {str(e)}")
        except Exception as e:
            print(f"Error generating game patterns: {str(e)}")
    
    async def get_game_rules(self, game_name: str) -> str:
        """
        Retrieve the rules for a specific game from the game_embeddings table
        
        Args:
            game_name: Name of the game to get rules for
            
        Returns:
            Rules text for the game
        """
        try:
            result = supabase.table("game_embeddings").select("content").eq("game_name", game_name).eq("content_type", "rules").execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]["content"]
            
            # If multiple parts, try to combine them
            result = supabase.table("game_embeddings").select("content").eq("game_name", game_name).like("content_type", "rules_part%").order("content_type").execute()
            
            if result.data and len(result.data) > 0:
                combined_rules = ""
                for part in result.data:
                    combined_rules += part["content"] + "\n\n"
                return combined_rules
            
            return "Game rules not found"
        except Exception as e:
            print(f"Error getting game rules: {str(e)}")
            return "Error retrieving game rules"
    
    async def get_common_outcomes(self, game_name: str, limit: int = 3) -> List[Dict]:
        """
        Get the most common outcomes for a game
        
        Args:
            game_name: Name of the game
            limit: Maximum number of outcomes to return
            
        Returns:
            List of common outcomes with their reasons and frequencies
        """
        try:
            # Get all outcomes for this game
            result = supabase.table("game_outcomes").select("*").eq("game_name", game_name).execute()
            
            # Sort manually by frequency
            if result.data:
                result.data.sort(key=lambda x: x.get("frequency", 0), reverse=True)
                
                # Limit to the requested number
                result.data = result.data[:limit]
            
            return result.data
        except Exception as e:
            print(f"Error getting common outcomes: {str(e)}")
            return []
    
    async def get_game_code(self, game_name: str) -> str:
        """
        Retrieve the implementation code for a specific game
        """
        try:
            result = supabase.table("game_embeddings").select("content").eq("game_name", game_name).eq("content_type", "code").execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]["content"]
            
            # If multiple parts, try to combine them
            result = supabase.table("game_embeddings").select("content").eq("game_name", game_name).like("content_type", "code_part%").order("content_type").execute()
            
            if result.data and len(result.data) > 0:
                combined_code = ""
                for part in result.data:
                    combined_code += part["content"] + "\n\n"
                return combined_code
            
            return "Game code not found"
        except Exception as e:
            print(f"Error getting game code: {str(e)}")
            return "Error retrieving game code"


# Example usage to test the GameStateManager
async def main():
    # Initialize the GameStateManager
    manager = GameStateManager()
    
    # Test retrieving game rules
    game_name = "Poker"
    rules = await manager.get_game_rules(game_name)
    print(f"Rules for {game_name}:")
    print(rules[:200] + "..." if len(rules) > 200 else rules)
    
    # Test starting a game session
    session_id = await manager.start_game_session(game_name)
    print(f"Started session: {session_id}")
    
    # Test storing a turn
    observation = "You have a pair of Kings. The pot is $100."
    await manager.store_turn(session_id, 0, observation)
    
    # Test updating a turn with an action
    action = "Raise to $50"
    reasoning = "I have a strong hand with a pair of Kings"
    await manager.update_turn_action(session_id, 1, action, reasoning)
    
    # Test ending a game session
    info = {"reason": "Won with a pair of Kings against a pair of Queens"}
    await manager.end_game_session(session_id, "win", info, 150.0)
    
    print("GameStateManager test complete")

if __name__ == "__main__":
    asyncio.run(main())
