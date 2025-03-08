import textarena as ta
import inspect

# Print available environment IDs
print("Available Environment IDs:")
try:
    # Try to access the environment registry
    if hasattr(ta, 'NAME_TO_ID_DICT'):
        for env_name in ta.NAME_TO_ID_DICT.keys():
            print(f"  - {env_name}")
    elif hasattr(ta, 'registry') and hasattr(ta.registry, 'NAME_TO_ID_DICT'):
        for env_name in ta.registry.NAME_TO_ID_DICT.keys():
            print(f"  - {env_name}")
    else:
        # Alternative approach: try to make each environment
        # List of common TextArena environment names to try
        env_names = [
            "Poker-v0", 
            "SimpleNegotiation-v0", 
            "Negotiation-v0",
            "SpellingBee-v0",
            "Tak-v0",
            "Wordle-v0",
            "Mastermind-v0"
        ]
        
        for env_name in env_names:
            try:
                env = ta.make(env_id=env_name)
                print(f"  - {env_name} (✓)")
                env.close()
            except Exception as e:
                print(f"  - {env_name} (✗) - {str(e)}")
except Exception as e:
    print(f"Error accessing environment registry: {e}")

# Inspect make_online function
print("\nInspecting ta.make_online function:")
if hasattr(ta, 'make_online'):
    sig = inspect.signature(ta.make_online)
    print(f"Signature: {sig}")
    
    # Get the docstring
    doc = ta.make_online.__doc__
    if doc:
        print("\nDocumentation:")
        print(doc)
    else:
        print("\nNo documentation available.")
else:
    print("make_online function not found in the textarena module.")

# Try to get the source code
try:
    source = inspect.getsource(ta.make_online)
    print("\nSource code:")
    print(source)
except Exception as e:
    print(f"\nCouldn't get source code: {e}")

print("\nRecommended usage:")
print("""
# For local testing:
env = ta.make(env_id="SimpleNegotiation-v0")  # Use a single environment ID

# For online testing:
env = ta.make_online(
    env_id="SimpleNegotiation-v0",  # Use a single environment ID or a list of valid IDs
    model_name="Your Model Name",
    model_description="Brief description of your model",
    email="your.email@example.com"
)
""") 