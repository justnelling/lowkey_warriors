import textarena as ta
from textarena_game_agent import RAGGameAgent
# Initialize agents
agents = {
    0: RAGGameAgent(),
    1: ta.agents.OpenRouterAgent(model_name="anthropic/claude-3.5-haiku"),
}

# Initialize environment from subset and wrap it
env = ta.make(env_id="SpellingBee-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "RAG", 1: "claude-3.5-haiku"},
)

async def play_game():
    env.reset(num_players=len(agents))
    done = False
    while not done:
        player_id, observation = env.get_observation()
        action = await agents[player_id](observation)
        done, info = env.step(action=action)
    rewards = env.close()
    return rewards

# Run the async function
import asyncio
rewards = asyncio.run(play_game())