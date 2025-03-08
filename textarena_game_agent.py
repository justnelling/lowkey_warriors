import os
import re
import json
import asyncio
from typing import Dict, Any, Tuple, Optional, List
from dotenv import load_dotenv

# Import the GameStateManager from RAG_gamestate
from RAG_gamestate import GameStateManager

# Load environment variables
load_dotenv()

class RAGGameAgent:
    """
    Agent that uses RAG to improve game playing by leveraging past experiences
    and game knowledge
    """
    
    def __init__(self, model_name: str = "claude-3-7-sonnet-20250219"):
        """Initialize the RAG Game Agent"""
        self.model_name = model_name
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
        known_games = ["Poker", "Negotiation", "SpellingBee", "Chess", "TicTacToe", "Battleship"]
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
            Dictionary with game rules, patterns, and outcomes
        """
        # Get the rules for this game
        rules = await self.state_manager.get_game_rules(game_name)
        
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
            "patterns": patterns,
            "outcomes": outcomes,
            "similar_observations": similar_observations
        }
    
    async def generate_action(self, observation: str, player_id: int) -> Tuple[str, str]:
        """
        Generate an action for the current game state using RAG
        
        Args:
            observation: Current game observation
            player_id: ID of the player (usually 0 for the agent)
            
        Returns:
            Tuple of (action, reasoning)
        """
        if not self.current_session_id:
            # Extract game name from observation
            game_name = await self._extract_game_name(observation)
            self.current_game_name = game_name
            self.current_session_id = await self.start_game(game_name)
        
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
        
        # Add patterns if available
        if context['patterns']:
            prompt += "\nRelevant Patterns:\n"
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
        Based on the game rules, current state, and past experiences, decide on the best action to take.
        
        Provide your response in this format:
        ACTION: [Your chosen action exactly as it should be sent to the game]
        REASONING: [Your step-by-step reasoning for this action]
        """
        
        # Generate response using Anthropic
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=0.7,
                system="You are an expert game player. Analyze the game state carefully and choose the optimal action.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            
            # Extract action and reasoning
            action_match = re.search(r'ACTION:\s*(.*?)(?:\n|$)', content)
            reasoning_match = re.search(r'REASONING:\s*(.*?)(?:\n\n|$)', content, re.DOTALL)
            
            action = action_match.group(1).strip() if action_match else content
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
            # Update the game state with the action and reasoning
            await self.state_manager.update_turn_action(
                self.current_session_id,
                self.current_turn,
                action,
                reasoning
            )
            
            return action, reasoning
        except Exception as e:
            print(f"Error generating action: {str(e)}")
            return "Error generating action", str(e)
    
    async def __call__(self, observation: str) -> str:
        """
        Process an observation and return an action
        
        Args:
            observation: Current game observation
            
        Returns:
            Action to take
        """
        # Try to extract player_id from the observation
        player_id_match = re.search(r'Player (\d+)', observation)
        if player_id_match:
            player_id = int(player_id_match.group(1))
        else:
            # Default to player 0
            player_id = 0
        
        self.player_id = player_id
        
        # If this is the first observation, extract game name and start session
        if not self.current_session_id:
            # Extract game name from observation
            game_name = await self._extract_game_name(observation)
            self.current_game_name = game_name
            self.current_session_id = await self.start_game(game_name)
        
        # Generate action and reasoning
        action, _ = await self.generate_action(observation, player_id)
        return action


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