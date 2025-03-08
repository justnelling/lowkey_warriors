import os
import re
import json
import asyncio
from typing import Dict, Any, Tuple, Optional, List
from dotenv import load_dotenv
from textarena.core import Agent

# Import the GameStateManager from RAG_gamestate
from RAG_gamestate import GameStateManager

# Load environment variables
load_dotenv()

class RAGGameAgent(Agent):
    """
    Agent that uses RAG to improve game playing by leveraging past experiences
    and game knowledge. Inherits from textarena.core.Agent for compatibility.
    """
    
    def __init__(self, model_name: str = "claude-3-7-sonnet-20250219", system_prompt: Optional[str] = None, max_tokens: int = 1500, temperature: float = 0.7, verbose: bool = False):
        """Initialize the RAG Game Agent"""
        super().__init__()
        self.model_name = model_name
        self.system_prompt = system_prompt or "You are an expert game player and programmer. Analyze the game state and code carefully to choose the optimal action."
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        
        self.state_manager = GameStateManager()
        self.current_session_id = None
        self.current_game_name = None
        self.current_turn = 0
        self.player_id = 0
        
        # Initialize Anthropic client
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            raise ImportError(
                "Anthropic package is required for RAGGameAgent. "
                "Install it with: pip install anthropic"
            )
    
    async def start_game(self, game_name: str, opponent: str = "unknown") -> str:
        """
        Start a new game session
        
        Args:
            game_name: Name of the game being played
            opponent: Identifier for the opponent
            
        Returns:
            session_id: UUID of the created game session
        """
        self.current_game_name = game_name
        self.current_turn = 0
        self.current_session_id = await self.state_manager.start_game_session(game_name, opponent)
        return self.current_session_id
    
    async def end_game(self, info: Dict[str, Any], rewards: Dict[int, float]):
        """
        End the current game session
        
        Args:
            info: Info dictionary from the environment, containing 'reason'
            rewards: Dictionary mapping player IDs to rewards
        """
        if self.current_session_id:
            # Determine outcome based on reward
            my_reward = rewards.get(self.player_id, 0)
            if my_reward > 0:
                outcome = "win"
            elif my_reward < 0:
                outcome = "loss"
            else:
                outcome = "draw"
                
            await self.state_manager.end_game_session(
                self.current_session_id, 
                outcome, 
                info, 
                my_reward
            )
            
            self.current_session_id = None
            self.current_game_name = None
            self.current_turn = 0
    
    async def _extract_game_name(self, observation: str) -> str:
        """
        Extract the game name from the observation
        
        Args:
            observation: The game observation text
            
        Returns:
            The extracted game name or a default
        """
        # Try different patterns to extract game name
        patterns = [
            r'playing (?:the )?(\w+(?:-\w+)*) game',  # "playing the Poker game" or "playing Poker game"
            r'Welcome to (?:the )?(\w+(?:-\w+)*)',     # "Welcome to the Poker" or "Welcome to Poker"
            r'\[GAME\] You are playing (\w+(?:-\w+)*)', # "[GAME] You are playing Poker"
            r'(\w+(?:-\w+)*)-v\d+',                    # "Poker-v0"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, observation, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # If no match found, look for known game names
        known_games = ["Poker", "Negotiation", "SpellingBee"]
        for game in known_games:
            if game.lower() in observation.lower():
                return game
        
        # Default to unknown
        return "UnknownGame"
    
    async def get_game_context(self, game_name: str, observation: str) -> Dict:
        """
        Get context information about a game to help the agent understand it
        
        Args:
            game_name: Name of the game to get context for
            observation: Current observation to find relevant patterns
            
        Returns:
            Dictionary with game rules, code, patterns, and outcomes
        """
        # Get the rules for this game
        rules = await self.state_manager.get_game_rules(game_name)
        
        # Get the implementation code for this game
        code = await self.state_manager.get_game_code(game_name)
        
        # Get relevant patterns for this observation
        patterns = await self.state_manager.retrieve_game_patterns(game_name, observation)
        
        # Get common outcomes for this game
        outcomes = await self.state_manager.get_common_outcomes(game_name)
        
        # Get similar past observations
        similar_observations = await self.state_manager.retrieve_similar_observations(
            observation,
            game_name,
            limit=3
        )
        
        return {
            "game_name": game_name,
            "rules": rules,
            "code": code,
            "patterns": patterns,
            "outcomes": outcomes,
            "similar_observations": similar_observations
        }
    
    async def _make_request(self, observation: str) -> str:
        """
        Make a single API request to Anthropic and return the generated action.
        This follows the pattern from AsyncAnthropicAgent.
        """
        # If this is the first observation, extract game name and start session
        if not self.current_session_id:
            # Extract game name from observation
            game_name = await self._extract_game_name(observation)
            self.current_game_name = game_name
            self.current_session_id = await self.start_game(game_name)
        
        # Try to extract player_id from the observation
        player_id_match = re.search(r'Player (\d+)', observation)
        if player_id_match:
            player_id = int(player_id_match.group(1))
        else:
            # Default to player 0
            player_id = 0
        
        self.player_id = player_id
        
        # Generate action and reasoning
        action, _ = await self.generate_action(observation, player_id)
        return action
    
    async def _retry_request(self, observation: str, retries: int = 3, delay: int = 5) -> str:
        """
        Attempt to make an API request with retries.
        This follows the pattern from AsyncAnthropicAgent.

        Args:
            observation (str): The input to process.
            retries (int): The number of attempts to try.
            delay (int): Seconds to wait between attempts.

        Raises:
            Exception: The last exception caught if all retries fail.
        """
        last_exception = None
        for attempt in range(1, retries + 1):
            try:
                response = await self._make_request(observation)
                if self.verbose:
                    print(f"\nObservation: {observation}\nResponse: {response}")
                return response
            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < retries:
                    await asyncio.sleep(delay)
        raise last_exception
    
    async def generate_action(self, observation: str, player_id: int) -> Tuple[str, str]:
        """
        Generate an action for the current game state using RAG
        
        Args:
            observation: Current game observation
            player_id: ID of the player (usually 0 for the agent)
            
        Returns:
            Tuple of (action, reasoning)
        """
        self.player_id = player_id
        self.current_turn += 1
        
        # Store the current observation (without action yet)
        await self.state_manager.store_turn(
            self.current_session_id,
            player_id,
            observation
        )
        
        # Get context for this game
        context = await self.get_game_context(self.current_game_name, observation)
        
        # Prepare the prompt with RAG context
        prompt = f"""
        You are playing the game: {self.current_game_name}
        
        Game Rules:
        {context['rules'][:3000] if len(context['rules']) > 3000 else context['rules']}
        
        Current Game State:
        {observation}
        
        """
        
        # Add code-based reasoning section
        prompt += f"""
        First, analyze the game mechanics and possible actions by thinking through the implementation:
        
        Game Implementation (for reference):
        ```python
        {context['code'][:2000] if len(context['code']) > 2000 else context['code']}
        ```
        
        Based on this implementation, break down:
        1. What actions are valid in the current state
        2. How the game processes these actions
        3. What the likely outcomes of different actions would be
        """
        
        # Add patterns if available
        if context['patterns']:
            prompt += "\nRelevant Patterns from Past Games:\n"
            for pattern in context['patterns']:
                prompt += f"""
                Pattern: {pattern['observation_pattern']}
                Recommended Action: {pattern['recommended_action']}
                Success Rate: {pattern['success_rate']}
                
                """
        
        # Add similar observations if available
        if context['similar_observations']:
            prompt += "\nSimilar Past Game States:\n"
            for i, obs in enumerate(context['similar_observations']):
                prompt += f"""
                Example {i+1}:
                Observation: {obs['observation'][:200]}...
                Action Taken: {obs['action']}
                Outcome: {obs.get('outcome', 'unknown')}
                
                """
        
        # Add common outcomes if available
        if context['outcomes']:
            prompt += "\nCommon Game Outcomes:\n"
            for outcome in context['outcomes']:
                prompt += f"- {outcome['outcome'].upper()}: {outcome['reason']} (seen {outcome['frequency']} times)\n"
        
        prompt += """
        Based on your analysis of the game mechanics, rules, current state, and past experiences:
        
        1. First, write a brief pseudocode or algorithm for how you'll approach this turn
        2. Then, decide on the best action to take
        
        Provide your response in this format:
        ALGORITHM:
        [Your step-by-step algorithm or pseudocode for approaching this game state]
        
        ACTION: [Your chosen action exactly as it should be sent to the game]
        
        REASONING: [Your detailed reasoning for this action]
        """
        
        # Generate response using Anthropic
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            
            # Extract algorithm, action and reasoning
            algorithm_match = re.search(r'ALGORITHM:(.*?)(?:ACTION:|$)', content, re.DOTALL)
            action_match = re.search(r'ACTION:\s*(.*?)(?:\n|$)', content)
            reasoning_match = re.search(r'REASONING:(.*?)(?:\n\n|$)', content, re.DOTALL)
            
            algorithm = algorithm_match.group(1).strip() if algorithm_match else ""
            action = action_match.group(1).strip() if action_match else content
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
            # Combine algorithm and reasoning for storage
            full_reasoning = f"ALGORITHM:\n{algorithm}\n\nREASONING:\n{reasoning}"
            
            # Update the game state with the action and reasoning
            await self.state_manager.update_turn_action(
                self.current_session_id,
                self.current_turn,
                action,
                full_reasoning
            )
            
            return action, full_reasoning
        except Exception as e:
            print(f"Error generating action: {str(e)}")
            return "Error generating action", str(e)
    
    async def __call__(self, observation: str) -> str:
        """
        Process an observation and return an action.
        This is the main entry point required by TextArena.
        
        Args:
            observation: Current game observation
            
        Returns:
            Action to take
        """
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received type: {type(observation)}")
        return await self._retry_request(observation)


# Example usage
async def main():
    # Initialize the RAG Game Agent
    agent = RAGGameAgent()
    
    # Example observation from a Poker game
    observation = "[GAME] You are Player 0. You are playing Poker. You have been dealt: [10♥, K♠]. The community cards are: [7♦, 2♣, Q♥]. Your opponent has raised to $20."
    
    # Get action
    action = await agent(observation)
    
    print(f"Observation: {observation}")
    print(f"Action: {action}")
    
    # Simulate game end
    info = {"reason": "You won with a pair of Kings"}
    rewards = {0: 50.0, 1: -50.0}
    await agent.end_game(info, rewards)
    
    # Try another game
    observation2 = "[GAME] You are Player 0. You are playing SpellingBee. The center letter is 'T'. The outer letters are: ['A', 'R', 'S', 'P', 'I', 'N']. Make as many words as possible using these letters."
    
    action2 = await agent(observation2)
    
    print(f"\nObservation: {observation2}")
    print(f"Action: {action2}")

if __name__ == "__main__":
    asyncio.run(main()) 