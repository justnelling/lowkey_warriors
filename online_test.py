import textarena as ta
import time
import asyncio
import traceback
from textarena_game_agent import RAGGameAgent
from dotenv import load_dotenv

load_dotenv()

model_name = "LOWKEY"
model_description = "lowkey warrior"
email = "lioneldeng.dev@gmail.com"

def run_game_with_async_agent():
    """Run a single game with an async agent in a synchronous environment"""
    # Initialize agent
    agent = RAGGameAgent()
    env = None
    
    try:
        # Create online environment
        print("Creating online environment...")
        env = ta.make_online(
            env_id=["SimpleNegotiation-v0", "Poker-v0", "SpellingBee-v0"], 
            model_name=model_name,
            model_description=model_description,
            email=email
        )
        env = ta.wrappers.LLMObservationWrapper(env=env)

        # Reset the environment
        print("Resetting environment...")
        env.reset(num_players=1)
        
        done = False
        info = {}
        
        while not done:
            try:
                print("\nWaiting for observation...")
                player_id, observation = env.get_observation()
                print(f"Received observation for player {player_id}: {observation[:100]}...")
                
                # Run the async agent in a new event loop
                print("Generating action...")
                action = asyncio.new_event_loop().run_until_complete(agent(observation))
                print(f"Generated action: {action[:100]}...")
                
                # Check if the game might be over before sending action
                if "game over" in observation.lower() or "match ended" in observation.lower():
                    print("Game appears to be over based on observation. Not sending action.")
                    done = True
                    info = {"reason": "Game over detected in observation"}
                    continue
                
                # Apply the action to the environment
                print("Sending action to environment...")
                done, info = env.step(action=action)
                print(f"Action result - done: {done}, info: {info}")
                print(f"Player {player_id} action: {action}")
            except Exception as e:
                error_str = str(e)
                print(f"Error during gameplay loop: {error_str}")
                
                # Check for specific error messages
                if "No active match" in error_str or "missing action" in error_str:
                    print("Game has ended on the server side. Exiting gameplay loop.")
                    done = True
                    info = {"reason": f"Server ended game: {error_str}"}
                elif "ClientConnection" in error_str or "WebSocket" in error_str or "connection" in error_str.lower():
                    print(f"Connection error during gameplay: {error_str}")
                    done = True
                    info = {"reason": f"Connection error: {error_str}"}
                else:
                    # Re-raise other errors
                    raise
        
        # Try to close the environment gracefully
        try:
            if env:
                print("Closing environment...")
                rewards = env.close()
                print(f"Game finished with rewards: {rewards}")
                print(f"Info: {info}")
        except Exception as e:
            print(f"Error closing environment: {e}")
        
        # End the game session in the RAG agent
        try:
            print("Ending game session in agent...")
            asyncio.new_event_loop().run_until_complete(agent.end_game(info, rewards if 'rewards' in locals() else {}))
            print("Game session ended in agent.")
        except Exception as e:
            print(f"Error ending game session in agent: {e}")
        
        return True
    except Exception as e:
        print(f"EXCEPTION: {e}")
        # Print stack trace for debugging
        traceback.print_exc()
        
        # Try to close the environment if it exists
        try:
            if env:
                env.close()
        except:
            pass
            
        return False

# Main loop
num_games = 10
retry_delay = 30

for i in range(num_games):
    print(f"\n--- Starting game {i+1}/{num_games} ---")
    
    success = run_game_with_async_agent()
    
    if success:
        print(f"Game {i+1} completed successfully. Waiting 10 seconds before next game...")
        time.sleep(10)
    else:
        print(f"Game {i+1} failed. Waiting {retry_delay} seconds before retry...")
        time.sleep(retry_delay)