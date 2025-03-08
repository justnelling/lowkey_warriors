import os
import re
import asyncio
import subprocess
import time
from typing import Dict, Any, Tuple, Optional, List

# Import the base RAGGameAgent
from textarena_game_agent import RAGGameAgent

# Import MCP components with correct paths
import sys
import json
import subprocess
import mcp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

class ImprovedRAGAgent(RAGGameAgent):
    """
    Enhanced RAG Game Agent with dictionary-based word validation 
    for Spelling Bee games using the MCP as a subprocess.
    """
    
    def __init__(self, model_name: str = "claude-3-7-sonnet-20250219", system_prompt: Optional[str] = None, 
                 max_tokens: int = 1500, temperature: float = 0.7, verbose: bool = False, mcp_path: Optional[str] = None):
        """Initialize the Improved RAG Game Agent with MCP capabilities"""
        super().__init__(model_name, system_prompt, max_tokens, temperature, verbose)
        
        # Initialize MCP related fields
        self.mcp_process = None
        self.mcp_path = mcp_path
        self.spelling_dictionary = None
        
        # MCP client session components
        self.mcp_session = None
        self.exit_stack = AsyncExitStack()
        self.stdio = None
        self.write = None
        
        self.game_state = {
            "allowed_letters": [],
            "submitted_words": []
        }
    
    async def start_mcp(self) -> bool:
        """
        Start the MCP subprocess and connect to it using the proper client
        
        Returns:
            bool: True if MCP is running and connected
        """
        if self.mcp_session is not None:
            if self.verbose:
                print("MCP is already running")
            return True
            
        if not self.mcp_path:
            if self.verbose:
                print("MCP path not provided, cannot start MCP")
            return False
            
        try:
            # Make sure the script path ends with .py
            if not self.mcp_path.endswith('.py'):
                if self.verbose:
                    print(f"MCP script path should end with .py: {self.mcp_path}")
                return False
                
            # Set up server parameters
            server_params = StdioServerParameters(
                command="python",
                args=[self.mcp_path],
                env=None
            )
            
            if self.verbose:
                print(f"Starting MCP server with: python {self.mcp_path}")
            
            # Create stdio transport
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            
            # Create and initialize session
            self.mcp_session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            await self.mcp_session.initialize()
            
            # Get available tools to verify connection
            response = await self.mcp_session.list_tools()
            if self.verbose:
                print(f"Connected to MCP server with tools: {[tool.name for tool in response.tools]}")
                
            return True
        except Exception as e:
            if self.verbose:
                print(f"Failed to start MCP: {e}")
            await self.stop_mcp()
            return False

    def is_mcp_running(self) -> bool:
        """
        Check if the MCP session is initialized
        
        Returns:
            bool: True if MCP session is active
        """
        return self.mcp_session is not None
    
    async def stop_mcp(self) -> None:
        """Stop the MCP session and clean up resources"""
        try:
            await self.exit_stack.aclose()
            self.mcp_session = None
            self.stdio = None
            self.write = None
            
            if self.verbose:
                print("Closed MCP session")
        except Exception as e:
            if self.verbose:
                print(f"Error stopping MCP: {e}")

    async def _extract_spelling_bee_info(self, observation: str) -> List[str]:
        """
        Extract Spelling Bee game information from the observation
        
        Args:
            observation: The game observation text
            
        Returns:
            List of allowed letters
        """
        # Try to find the allowed letters section
        allowed_letters_match = re.search(r'Allowed Letters: ([a-z ]+)', observation, re.IGNORECASE)
        
        if allowed_letters_match:
            letters_str = allowed_letters_match.group(1).strip()
            allowed_letters = [letter.lower() for letter in letters_str.split()]
            if self.verbose:
                print(f"Extracted allowed letters from explicit mention: {allowed_letters}")
            return allowed_letters
        
        # Try to extract from np.str_ format
        np_str_match = re.findall(r"np\.str_\('([a-z])'\)", observation, re.IGNORECASE)
        if np_str_match:
            allowed_letters = [letter.lower() for letter in np_str_match]
            if self.verbose:
                print(f"Extracted allowed letters from np.str_ format: {allowed_letters}")
            return allowed_letters
        
        # Alternate format: they may be listed differently
        letters_match = re.findall(r'\b([a-z])\b', observation.lower())
        if letters_match:
            # Filter to ensure we only get unique lowercase letters
            allowed_letters = list(set(letters_match))
            if self.verbose:
                print(f"Extracted allowed letters from general text: {allowed_letters}")
            return allowed_letters
        
        return []

    async def start_game(self, game_name: str, opponent: str = "unknown") -> str:
        """
        Start a new game session
        
        Args:
            game_name: Name of the game being played
            opponent: Identifier for the opponent
            
        Returns:
            session_id: UUID of the created game session
        """
        session_id = await super().start_game(game_name, opponent)
        
        # For Spelling Bee, load the EnglishDictionary directly
        # Normalize the game name to handle both "Spelling Bee" and "spellingbee" formats
        normalized_game_name = game_name.lower().replace(" ", "")
        
        if normalized_game_name == "spellingbee":
            try:
                # Start MCP if needed
                if self.mcp_path and not self.is_mcp_running():
                    mcp_started = await self.start_mcp()
                    if self.verbose:
                        print(f"Started MCP for SpellingBee game: {mcp_started}")
                
                from textarena.utils.word_lists import EnglishDictionary
                self.spelling_dictionary = EnglishDictionary(keep_proper_nouns=False, include_nltk=True)
                if self.verbose:
                    print("Loaded EnglishDictionary for Spelling Bee")
            except ImportError as e:
                print(f"Could not load EnglishDictionary: {e}")
        
        return session_id
    
    async def end_game(self, info: Dict[str, Any], rewards: Dict[int, float]):
        """
        End the current game session
        
        Args:
            info: Info dictionary from the environment, containing 'reason'
            rewards: Dictionary mapping player IDs to rewards
        """
        await super().end_game(info, rewards)
        
        # Stop MCP if it's running
        await self.stop_mcp()
        
        # Reset local game state
        self.game_state = {
            "allowed_letters": [],
            "submitted_words": []
        }
        self.spelling_dictionary = None
    
    async def initialize_game(self, allowed_letters: List[str]) -> Dict[str, Any]:
        """
        Initialize the Spelling Bee game state
        
        Args:
            allowed_letters: Letters that can be used in the game
            
        Returns:
            The initialized game state
        """
        # Update local state
        self.game_state = {
            "allowed_letters": [letter.lower() for letter in allowed_letters],
            "submitted_words": []
        }
        
        # Also update the MCP's state if it's running
        if self.is_mcp_running():
            try:
                # Call the MCP to initialize the game
                result = await self.call_mcp_tool("initialize_game", {
                    "allowed_letters": [letter.lower() for letter in allowed_letters]
                })
                if self.verbose:
                    print(f"MCP initialize_game result: {result}")
            except Exception as e:
                if self.verbose:
                    print(f"Error calling MCP initialize_game: {e}")
        
        return self.game_state
    
    async def check_word(self, word: str) -> Dict[str, Any]:
        """
        Check if a word is valid for the Spelling Bee game.
        
        Args:
            word: The word to check
            
        Returns:
            Information about whether the word is valid, along with suggestions if invalid
        """
        # Strip brackets if present
        clean_word = word
        if word.startswith('[') and word.endswith(']'):
            clean_word = word[1:-1]
        clean_word = clean_word.strip().lower()
        
        # First try using the MCP if available
        if self.is_mcp_running():
            try:
                # Call the MCP to check the word
                result = await self.call_mcp_tool("check_word", {"word": clean_word})
                if self.verbose:
                    print(f"MCP check_word result: {result}")
                    
                if "error" not in result:
                    return result
                else:
                    print(f"Error from MCP: {result.get('error')}")
                    # Fall back to internal dictionary if MCP fails
            except Exception as e:
                if self.verbose:
                    print(f"Error calling MCP check_word: {e}")
                # Fall back to internal dictionary
        
        # Use the internal dictionary as fallback
        if self.verbose:
            print("Falling back to internal dictionary check")
        
        # Original internal dictionary check logic
        if self.spelling_dictionary is None:
            return {
                "is_valid": False,
                "reason": "Dictionary not loaded",
                "suggestions": []
            }
            
        # Get allowed letters from game state
        allowed_letters = self.game_state.get("allowed_letters", [])
        
        result = {
            "word": clean_word,
            "is_valid": False,
            "reason": None,
            "suggestions": []
        }
        
        # Check if the word exists in our dictionary
        if not self.spelling_dictionary.is_english_word(clean_word):
            result["reason"] = f"'{clean_word}' is not in our dictionary."
            return result
        
        # Check if word only uses allowed letters
        if allowed_letters and not all(letter in allowed_letters for letter in clean_word):
            invalid_chars = [letter for letter in clean_word if letter not in allowed_letters]
            result["reason"] = f"Word contains letters that aren't allowed: {invalid_chars}"
            return result
        
        # Word is valid
        result["is_valid"] = True
        return result
    
    async def format_word_for_submission(self, word: str) -> str:
        """
        Format a word for submission by wrapping it in square brackets.
        
        Args:
            word: The word to format
            
        Returns:
            The word formatted for submission
        """
        # First try using the MCP if available
        if self.is_mcp_running():
            try:
                # Call the MCP to format the word
                result = await self.call_mcp_tool("format_word_for_submission", {"word": word})
                if self.verbose:
                    print(f"MCP format_word_for_submission result: {result}")
                    
                if isinstance(result, str):
                    return result
                # Fall back to manual formatting if MCP fails
            except Exception as e:
                if self.verbose:
                    print(f"Error calling MCP format_word_for_submission: {e}")
        
        # Clean the word and wrap in brackets
        clean_word = word.strip().lower()
        return f"[{clean_word}]"
    
    async def generate_action(self, observation: str, player_id: int) -> Tuple[str, str]:
        """
        Generate an action for the current game state using RAG and dictionary support
        
        Args:
            observation: Current game observation
            player_id: ID of the player (usually 0 for the agent)
            
        Returns:
            Tuple of (action, reasoning)
        """
        # First, update player_id, turn, and store the observation
        self.player_id = player_id
        self.current_turn += 1
        
        # Store the current observation (without action yet)
        await self.state_manager.store_turn(
            self.current_session_id,
            player_id,
            observation
        )
        
        # Normalize game name for consistent comparison
        normalized_game_name = self.current_game_name.lower().replace(" ", "") if self.current_game_name else ""
        
        # Special handling for Spelling Bee game
        if normalized_game_name == "spellingbee":
            # Check if this is the first turn, if so initialize game
            if self.current_turn == 1 or not self.game_state.get("allowed_letters"):
                allowed_letters = await self._extract_spelling_bee_info(observation)
                if allowed_letters:
                    await self.initialize_game(allowed_letters)
                
            # Try to find and submit the longest valid word
            if self.is_mcp_running():
                try:
                    # Use MCP to find valid words sorted by length
                    word_results = await self.call_mcp_tool("find_valid_words", {"max_suggestions": 50})
                    if self.verbose:
                        print(f"MCP find_valid_words result: {word_results}")
                    
                    # Find the longest word from the suggestions
                    best_word = ""
                    if "words_by_length" in word_results:
                        words_by_length = word_results["words_by_length"]
                        # Find the longest length that has words
                        for length in sorted([int(k) for k in words_by_length.keys()], reverse=True):
                            words = words_by_length[str(length)]
                            if words:
                                best_word = words[0].lower()  # Ensure lowercase
                                break
                    
                    if best_word:
                        # Submit the word in the correct format with brackets
                        action = f"[{best_word}]"
                        reasoning = f"Submitting '{best_word}' as it's a valid word with {len(best_word)} letters using the allowed letters."
                        
                        # Update the turn with action and reasoning
                        await self.state_manager.update_turn_action(
                            self.current_session_id,
                            self.current_turn,
                            action,
                            reasoning
                        )
                        
                        return action, reasoning
                    else:
                        # If no valid words found, fall back to the base RAG approach
                        if self.verbose:
                            print("No valid words found, falling back to base RAG approach")
                except Exception as e:
                    if self.verbose:
                        print(f"Error calling MCP find_valid_words: {e}")
            
            # If MCP failed or returned no words, fallback to base RAG approach
        
        # For other games or if the special handling didn't return, fall back to the base RAG approach
        # But first, decrement the turn counter since it will be incremented again in the super method
        self.current_turn -= 1
        
        # Call the parent class's generate_action method
        action, reasoning = await super().generate_action(observation, player_id)
        
        # If this is Spelling Bee, ensure the response is properly formatted with brackets
        if normalized_game_name == "spellingbee":
            # Make sure the action contains a word in brackets
            if not (action.startswith('[') and action.endswith(']')):
                # Try to find a word in the action
                word_match = re.search(r'\b([a-z]{3,})\b', action.lower())
                if word_match:
                    word = word_match.group(1)
                    
                    # Check if the word is valid
                    check_result = await self.check_word(word)
                    
                    if check_result.get("is_valid", False):
                        new_action = f"[{word.lower()}]"
                        new_reasoning = f"Formatted '{word}' for submission: {new_action}"
                        
                        # Update the turn with formatted action and reasoning
                        await self.state_manager.update_turn_action(
                            self.current_session_id,
                            self.current_turn,
                            new_action,
                            new_reasoning
                        )
                        
                        return new_action, new_reasoning
                    else:
                        # Word not valid, try to find a valid word using allowed letters
                        if self.spelling_dictionary and self.game_state.get("allowed_letters"):
                            allowed_letters = self.game_state.get("allowed_letters")
                            valid_words = []
                            
                            # Simple search for a valid word
                            for dict_word in self.spelling_dictionary.get_all_words():
                                if (len(dict_word) >= 4 and
                                    all(letter in allowed_letters for letter in dict_word)):
                                    valid_words.append(dict_word)
                                    if len(valid_words) >= 10:
                                        break
                            
                            if valid_words:
                                # Sort by length and take the longest
                                valid_words.sort(key=len, reverse=True)
                                word = valid_words[0].lower()
                                new_action = f"[{word}]"
                                new_reasoning = f"Submitting valid word '{word}' instead of invalid word."
                                return new_action, new_reasoning
                
                # If we couldn't extract a valid word, format the original response with brackets
                # but only if it's short enough to potentially be a word
                if len(action) <= 15 and action.isalpha():
                    action = f"[{action.lower()}]"
        
        return action, reasoning

    def get_fallback_word(self) -> str:
        """Get a fallback word when no optimal word can be found"""
        allowed_letters = self.game_state.get("allowed_letters", [])
        if not allowed_letters:
            return "test"  # Absolute fallback
        
        # Try to construct a valid word from the allowed letters
        if len(allowed_letters) >= 4:
            return ''.join(allowed_letters[:4])
        else:
            # If we have fewer than 4 letters, repeat some to reach minimum length
            return ''.join(allowed_letters) + allowed_letters[0] * (4 - len(allowed_letters))

    async def call_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool using the proper async MCP client
        
        Args:
            tool_name: Name of the tool to call
            params: Parameters to pass to the tool
            
        Returns:
            The result from the MCP tool call
        """
        # Ensure MCP is running
        if not self.is_mcp_running():
            if self.verbose:
                print(f"MCP not running, attempting to start for {tool_name}")
            mcp_started = await self.start_mcp()
            if not mcp_started:
                return {"error": "Failed to start MCP"}
        
        try:
            if self.verbose:
                print(f"Calling MCP tool: {tool_name} with params: {params}")
            
            # Call the tool using the MCP session
            result = await self.mcp_session.call_tool(tool_name, params)
            
            # Parse the nested response structure
            if hasattr(result, "content") and isinstance(result.content, list) and len(result.content) > 0:
                content_item = result.content[0]
                if hasattr(content_item, "text") and content_item.text:
                    try:
                        # Parse the JSON string inside the text field
                        parsed_data = json.loads(content_item.text)
                        if self.verbose:
                            print(f"Successfully parsed MCP response: {parsed_data}")
                        return parsed_data
                    except json.JSONDecodeError:
                        if self.verbose:
                            print(f"Failed to parse JSON in MCP response: {content_item.text}")
                        return {"content": content_item.text}
            
            # Convert result to dictionary if needed
            if hasattr(result, "model_dump"):
                return result.model_dump()
            elif hasattr(result, "dict"):
                return result.dict()
            else:
                # Convert the result to a dictionary format
                return {"content": str(result.content) if hasattr(result, "content") else str(result)}
                
        except Exception as e:
            if self.verbose:
                print(f"Error calling MCP tool: {e}")
            return {"error": f"Error calling MCP tool: {e}"}

# Example usage
async def main():
    # Initialize the Improved RAG Game Agent with MCP path
    mcp_path = "spelling_mcp.py"  # Update this to your actual MCP path
    agent = ImprovedRAGAgent(verbose=True, mcp_path=mcp_path)
    
    # Check if MCP is running before starting
    print(f"MCP running before start: {agent.is_mcp_running()}")
    
    # Start the game which should start MCP
    await agent.start_game("SpellingBee")
    
    # Check if MCP is running after starting
    print(f"MCP running after game start: {agent.is_mcp_running()}")
    
    # Example observation from a Spelling Bee game
    observation = "[GAME] You are Player 0. You are playing SpellingBee. The center letter is 'T'. The outer letters are: ['A', 'R', 'S', 'P', 'I', 'N']. Make as many words as possible using these letters."
    
    # Get action
    action = await agent(observation)
    
    print(f"Observation: {observation}")
    print(f"Action: {action}")
    
    # Test a word submission
    observation2 = "Is the word 'TRAIN' valid?"
    action2 = await agent(observation2)
    
    print(f"\nObservation: {observation2}")
    print(f"Action: {action2}")
    
    # Test an invalid word
    observation3 = "Check the word 'DINOSAUR'"
    action3 = await agent(observation3)
    
    print(f"\nObservation: {observation3}")
    print(f"Action: {action3}")
    
    # Clean up
    info = {"reason": "Testing complete"}
    rewards = {0: 100.0}
    await agent.end_game(info, rewards)
    
    # Check if MCP is running after ending the game
    print(f"MCP running after game end: {agent.is_mcp_running()}")

if __name__ == "__main__":
    asyncio.run(main()) 