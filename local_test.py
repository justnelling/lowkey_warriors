from textarena_game_agent import RAGGameAgent
import textarena as ta
from dotenv import load_dotenv
import asyncio

load_dotenv()

# Initialize agents
agents = {
    # 0: ta.agents.OpenRouterAgent(model_name="GPT-4o-mini"),
    0: ta.agents.OpenAIAgent(model_name="gpt-4"),
    1: RAGGameAgent(),
}

async def run_game():
    # Initialize environment from subset and wrap it
    env = ta.make(env_id="SimpleNegotiation-v0")
    env = ta.wrappers.LLMObservationWrapper(env=env)
    env = ta.wrappers.SimpleRenderWrapper(
        env=env,
        player_names={0: "GPT-4", 1: "LOWKEY"},
    )

    # Reset the environment with the number of players
    env.reset(num_players=len(agents))
    
    done = False
    while not done:
        player_id, observation = env.get_observation()
        
        # Handle different agent types
        if player_id == 1:  # RAGGameAgent
            action = await agents[player_id](observation)
        else:  # Regular agent
            action = agents[player_id](observation)
            
        done, info = env.step(action=action)
        print(f"Player {player_id} action: {action}")
    
    rewards = env.close()
    print(f"Game finished with rewards: {rewards}")
    
    # If the game is done, end the RAG agent's session
    if isinstance(agents[1], RAGGameAgent):
        await agents[1].end_game(info, rewards)
    
    return rewards

if __name__ == "__main__":
    asyncio.run(run_game())