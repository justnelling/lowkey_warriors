from improved_rag_agent import ImprovedRAGAgent
from textarena_game_agent import RAGGameAgent
import textarena as ta
from dotenv import load_dotenv
import asyncio

load_dotenv()

# Initialize agents
agents = {
    0: ImprovedRAGAgent(verbose=True, mcp_path="spelling_mcp.py"),
    1: ImprovedRAGAgent(verbose=True, mcp_path="spelling_mcp.py"),
}

async def run_game():
    # Check if MCP is running for each agent before starting
    for player_id, agent in agents.items():
        if isinstance(agent, ImprovedRAGAgent):
            print(f"Player {player_id} MCP running before game: {agent.is_mcp_running()}")

    # Initialize environment from subset and wrap it
    env = ta.make(env_id="SpellingBee-v0")
    env = ta.wrappers.LLMObservationWrapper(env=env)
    env = ta.wrappers.SimpleRenderWrapper(
        env=env,
        player_names={0: "LOWKEY-I1", 1: "LOWKEY-I2"},
    )

    # Start the game sessions for RAG agents
    for player_id, agent in agents.items():
        if isinstance(agent, (RAGGameAgent, ImprovedRAGAgent)):
            await agent.start_game("SpellingBee")
            if isinstance(agent, ImprovedRAGAgent):
                print(f"Player {player_id} MCP running after start_game: {agent.is_mcp_running()}")

    env.reset(num_players=len(agents))
    done = False
    step_count = 0  # Manually track step count
    
    while not done:
        player_id, observation = env.get_observation()
        
        # Periodic MCP status check (every 5 turns)
        if step_count % 5 == 0 and isinstance(agents[player_id], ImprovedRAGAgent):
            print(f"Turn {step_count}: Player {player_id} MCP running: {agents[player_id].is_mcp_running()}")
        
        # Handle different agent types
        if isinstance(agents[player_id], (RAGGameAgent, ImprovedRAGAgent)):  # RAGGameAgent
            action = await agents[player_id](observation)
        else:  # Regular agent
            action = agents[player_id](observation)
            
        done, info = env.step(action=action)
        print(f"Player {player_id} action: {action}")
        step_count += 1  # Increment step count after each step
    
    rewards = env.close()
    print(f"Game finished with rewards: {rewards}")
    
    # If the game is done, end all RAG agent sessions with better error handling
    for player_id, agent in agents.items():
        if isinstance(agent, (RAGGameAgent, ImprovedRAGAgent)):
            try:
                await agent.end_game(info, rewards)
                if isinstance(agent, ImprovedRAGAgent):
                    print(f"Player {player_id} MCP running after end_game: {agent.is_mcp_running()}")
            except Exception as e:
                print(f"Error ending game for player {player_id}: {e}")
    
    return rewards

if __name__ == "__main__":
    asyncio.run(run_game())