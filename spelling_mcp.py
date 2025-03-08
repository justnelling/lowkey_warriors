#!/usr/bin/env python3
from mcp.server.fastmcp import FastMCP, Context
from typing import Dict, List, Any, Optional
from textarena.utils.word_lists import EnglishDictionary
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("spelling-mcp")

# Create MCP server
mcp = FastMCP("Spelling Bee Dictionary")

# Store game state globally
GAME_STATE = {
    "allowed_letters": [],
    "submitted_words": []
}

# Initialize dictionary
try:
    dictionary = EnglishDictionary(keep_proper_nouns=False, include_nltk=True)
    logger.info("Successfully loaded English Dictionary")
except Exception as e:
    logger.error(f"Failed to load dictionary: {e}")
    dictionary = None

@mcp.resource("spellingbee://game-state")
def get_game_state() -> Dict[str, Any]:
    """Get the current state of the Spelling Bee game."""
    logger.info(f"Retrieved game state: {GAME_STATE}")
    return GAME_STATE

@mcp.tool()
def initialize_game(allowed_letters: List[str]) -> Dict[str, Any]:
    """
    Initialize or reset the Spelling Bee game.
    
    Args:
        allowed_letters: The list of letters that can be used to form words
        
    Returns:
        The initialized game state
    """
    global GAME_STATE
    
    # Ensure all letters are lowercase
    cleaned_letters = [letter.lower() for letter in allowed_letters]
    
    GAME_STATE = {
        "allowed_letters": cleaned_letters,
        "submitted_words": []
    }
    
    logger.info(f"Game initialized with allowed_letters={cleaned_letters}")
    return GAME_STATE

@mcp.tool()
def check_word(word: str, ctx: Context) -> Dict[str, Any]:
    """
    Check if a word is valid for the Spelling Bee game.
    
    Args:
        word: The word to check
        ctx: The MCP context
        
    Returns:
        Information about whether the word is valid, along with suggestions if invalid
    """
    # Clean the input word, removing brackets if present
    if word.startswith('[') and word.endswith(']'):
        word = word[1:-1]
    word = word.strip().lower()
    
    # Get allowed letters from game state
    allowed_letters = GAME_STATE.get("allowed_letters", [])
    
    ctx.info(f"Checking word: {word}")
    logger.info(f"Checking word '{word}' with allowed_letters={allowed_letters}")
    
    result = {
        "word": word,
        "is_valid": False,
        "reason": None,
        "suggestions": [],
        "length": len(word)
    }
    
    # Check if the dictionary is loaded
    if dictionary is None:
        result["reason"] = "Dictionary not loaded"
        return result
    
    # Check if the word exists in our dictionary
    if not dictionary.is_english_word(word):
        result["reason"] = f"'{word}' is not in our dictionary."
        # Generate suggestions
        result["suggestions"] = find_similar_valid_words(word, allowed_letters)
        return result
    
    # Check if word only uses allowed letters
    if allowed_letters and not all(letter in allowed_letters for letter in word):
        invalid_chars = [letter for letter in word if letter not in allowed_letters]
        result["reason"] = f"Word contains letters that aren't allowed: {invalid_chars}"
        return result
    
    # Word is valid
    result["is_valid"] = True
    
    # Track submitted word
    if word not in GAME_STATE["submitted_words"]:
        GAME_STATE["submitted_words"].append(word)
        ctx.info(f"Added word: {word}")
    
    return result

@mcp.tool()
def find_valid_words(max_suggestions: int = 10) -> Dict[str, Any]:
    """
    Find valid words for the current game using allowed letters.
    Returns words organized by length.
    
    Args:
        max_suggestions: Maximum number of suggestions to return per length
        
    Returns:
        Dictionary with word suggestions by length
    """
    allowed_letters = GAME_STATE.get("allowed_letters", [])
    
    if not allowed_letters:
        logger.warning("No allowed letters set for find_valid_words")
        return {
            "error": "No allowed letters set",
            "words_by_length": {}
        }
    
    if dictionary is None:
        logger.error("Dictionary not loaded for find_valid_words")
        return {
            "error": "Dictionary not loaded",
            "words_by_length": {}
        }
    
    logger.info(f"Finding valid words with letters: {allowed_letters}")
    all_possible_words = []
    for word in dictionary.get_all_words():
        # Check if word only uses allowed letters
        if all(letter in allowed_letters for letter in word):
            all_possible_words.append(word)
    
    # Sort words by length (longest first)
    all_possible_words.sort(key=len, reverse=True)
    
    # Organize words by length
    words_by_length = {}
    for word in all_possible_words:
        length = len(word)
        if length not in words_by_length:
            words_by_length[length] = []
        
        if len(words_by_length[length]) < max_suggestions:
            words_by_length[length].append(word)
    
    # Convert dictionary keys to strings for JSON serialization
    result = {
        "total_words_found": len(all_possible_words),
        "words_by_length": {str(k): v for k, v in words_by_length.items()}
    }
    
    logger.info(f"Found {len(all_possible_words)} valid words")
    return result

@mcp.tool()
def format_word_for_submission(word: str) -> str:
    """
    Format a word for submission by wrapping it in square brackets.
    
    Args:
        word: The word to format
        
    Returns:
        The word formatted for submission
    """
    # Clean the word and wrap in brackets
    clean_word = word.strip().lower()
    formatted = f"[{clean_word}]"
    logger.info(f"Formatted '{word}' to '{formatted}'")
    return formatted

def find_similar_valid_words(word: str, allowed_letters: List[str]) -> List[str]:
    """
    Find similar valid words to suggest when a word is invalid.
    
    Args:
        word: The invalid word
        allowed_letters: The list of allowed letters
        
    Returns:
        A list of valid word suggestions
    """
    if not allowed_letters or dictionary is None:
        return []
        
    suggestions = []
    prefix = word[:2] if len(word) >= 2 else word
    
    for dict_word in dictionary.get_all_words():
        if (dict_word.startswith(prefix) and all(letter in allowed_letters for letter in dict_word)):
            suggestions.append(dict_word)
            if len(suggestions) >= 5:  # Limit to 5 suggestions
                break
    
    # If we couldn't find similar words with the same prefix,
    # just suggest some valid words with the allowed letters
    if not suggestions:
        for dict_word in dictionary.get_all_words():
            if all(letter in allowed_letters for letter in dict_word):
                suggestions.append(dict_word)
                if len(suggestions) >= 5:
                    break
    
    return suggestions

if __name__ == "__main__":
    # This allows the MCP server to run directly when executed as a script
    logger.info("Starting TextArena Spelling Bee MCP server...")
    mcp.run()
