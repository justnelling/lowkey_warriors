import textarena as ta
from dotenv import load_dotenv

load_dotenv()

env = ta.make(env_id="Tak-v0-easy")

# Wrap the environment for easier observation handling
env = ta.wrappers.LLMObservationWrapper(env=env)

# Wrap the environment for pretty rendering
env = ta.wrappers.PrettyRenderWrapper(env=env)

# initalize agents
agents = {
    0: ta.basic_agents.OpenRouterAgent(model_name="gpt-4o"),
    1: ta.basic_agents.OpenRouterAgent(model_name="gpt-4o-mini")
}

# reset the environment to start a new game
observations = env.reset(seed=490)

# Game loop
done = False
while not done:

    # Get the current player id and observation
    player_id, observation = env.get_observation()

    # Agent decides on an action based on the observation
    action = agents[player_id](observation)

    # Execute the action in the environment
    rewards, truncated, terminated, info = env.step(action=action)

    # Check if the game has ended
    done = terminated or truncated

    # Optionally render the environment to see the current state
    env.render()

    if done:
        break

# Finally, print the game results
for player_id, agent in agents.items():
    print(f"&#123;agent.model_name&#125;: &#123;rewards[player_id]&#125;")
print(f"Reason: &#123;info['reason']&#125;")