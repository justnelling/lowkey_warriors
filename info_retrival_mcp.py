import os
import smithery
from supabase import create_client, Client

class KnowledgeGraphMCP:
    def __init__(self):
        self.url = smithery.create_smithery_url(
            "wss://server.smithery.ai/e2b/ws",
            {
                "e2bApiKey": os.environ["E2B_API_KEY"],
                "anthropicApiKey": os.environ["ANTHROPIC_API_KEY"],
            },
        )
        self.rules = self.load_game_rules()
        self.mcp_client = self.connect_to_mcp()

    def connect_to_mcp(self):
        return smithery.create_smithery_url("https://smithery.ai/server/@itseasy21/mcp-knowledge-graph")

    def connect_to_supabase(self) -> Client:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
        return create_client(url, key)
    
    def learn_from_game(self, game_data):
        # Add observation based on the game data
        self.mcp_client.add_observation(game_data)

    def analyze_game_outcome(self, game_data):
        #Retrieve data related to the game outcome
        outcome_data = self.mcp_client.read_graph()
        #or use search_nodes if needed
        #Analyze the outcome data to understand why it won/lost
        return outcome_data

    def read_vector_table(self):
        response = self.supabase_client.table("game_embeddings").select("*").execute()
        return response.data
