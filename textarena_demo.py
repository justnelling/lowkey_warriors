import textarena as ta
from dotenv import load_dotenv

load_dotenv()

# Initialize agents
agents = {
    0: ta.agents.OpenRouterAgent(model_name="GPT-4o-mini"),
    1: ta.agents.OpenRouterAgent(model_name="anthropic/claude-3.5-haiku"),
}

# Initialize environment from subset and wrap it
env = ta.make(env_id="SpellingBee-v0") # can pass list to this, make_online if we're submitting for it to play online 
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "GPT-4o-mini", 1: "claude-3.5-haiku"},
)

env.reset(num_players=len(agents))
done = False
while not done:
    player_id, observation = env.get_observation() # observation will be a string that we can pass into LLM for next move
    action = agents[player_id](observation) # action has to be a single string too
    done, info = env.step(action=action) # done = boolean, info will have a .reason key that will give explanation for why the agent lost / won 
rewards = env.close()


'''
3 environments released now as public set:

"SimpleNegotiation-v0", 
'''