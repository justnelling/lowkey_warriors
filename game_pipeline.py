from typing import List, Dict, Optional, Union
import json
from dataclasses import dataclass
from enum import Enum
import anthropic
import openai
from datetime import datetime
import psycopg2
from psycopg2.extras import Json
import numpy as np
from dotenv import load_dotenv
import os
import asyncio

'''
logic here: 

new game
    load into RAG pipeline
    
    look for games with similar rulesets

    analyze past strategies / outcomes

    synthesize new strategy for this game based on:
        - what it knows from before in textarena (RAG) 
        - what other games its come across online (MCP?)
'''

# --- Core Data Structures ---

class ActionType(Enum):
    POSITIONAL = "positional"
    TEXT_BASED = "text_based"
    LOGICAL = "logical"
    PERSUASION = "persuasion"
    RESOURCE_MANAGEMENT = "resource_management"

@dataclass
class GameAction:
    action_name: str
    state_before: str
    state_after: str
    result: str # 'success', 'failure', etc.

@dataclass
class GameContext:
    game_name: str
    current_state: str
    action_type: ActionType
    constraints: List[str]
    available_actions: List[str]
    action_history: List[GameAction] = None

@dataclass
class Strategy:
    context: GameContext
    core_concepts: List[str]
    action_pattern: str
    expected_outcome: str
    success_rate: float
    action_history: List[GameAction] = None

# --- Vector Store Implementation ---

class VectorStore:
    def __init__(self):
        load_dotenv()
        self.conn = None
        # Remove unused client instances - we'll create them when needed
        self._connect()

    def _connect(self):
        """Establish database connection"""
        try:
            if not self.conn or self.conn.closed:
                self.conn = psycopg2.connect(
                    dbname=os.getenv("POSTGRES_DB"),
                    user=os.getenv("POSTGRES_USER"),
                    password=os.getenv("POSTGRES_PASSWORD"),
                    host=os.getenv("POSTGRES_HOST", "localhost")
                )
        except Exception as e:
            print(f"Warning: Database connection failed: {e}")
            self.conn = None

    def close(self):
        """Close the database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()

    def setup_database(self):
        """Set up the database schema and required extensions"""
        if not self.conn:
            print("Database not connected - skipping setup")
            return

        try:
            with self.conn.cursor() as cur:
                # Create vector extension first
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.conn.commit()
                
                # Create the table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS game_strategies (
                        id SERIAL PRIMARY KEY,
                        game_name TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        context TEXT NOT NULL,
                        core_concepts JSONB NOT NULL,
                        action_pattern TEXT NOT NULL,
                        constraints JSONB,
                        success_metrics JSONB,
                        action_history JSONB,
                        context_embedding vector(1536),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                self.conn.commit()

                # Create the index
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS game_strategies_embedding_idx 
                    ON game_strategies 
                    USING hnsw (context_embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)
                self.conn.commit()
                print("Database setup completed successfully")
        except Exception as e:
            print(f"Error setting up database: {e}")
            print("Please ensure PostgreSQL is running and the vector extension is available")
            raise

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI's API"""
        try:
            # Use a consistent approach for client creation
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            # Convert the embedding to a list explicitly
            return list(response.data[0].embedding)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * 1536  # OpenAI's text-embedding-ada-002 has 1536 dimensions

    async def _ingest_new_game(self, game_context: GameContext) -> None:
        '''
        Ingest a new game and its ruleset into the vector database
        '''

        # create a comprehensive game description for embedding
        game_description = f"""
        Game Name: {game_context.game_name}
        Current State: {game_context.current_state}
        Action Type: {game_context.action_type.value}
        Constraints: {', '.join(game_context.constraints)}
        Available Actions: {', '.join(game_context.available_actions)}"""

        embedding = await self._generate_embedding(game_description)

        action_history_json = []
        if game_context.action_history:
            action_history_json = [
                {
                    'action_name': action.action_name,
                    'state_before': action.state_before,
                    'state_after': action.state_after,
                    'result': action.result
                }
                for action in game_context.action_history
            ]

        # prepare data for db insertion
        with self.conn.cursor() as cur:
            cur.execute("""
            INSERT INTO game_strategies (
                game_name,
                action_type,
                context,
                core_concepts,
                action_pattern,
                constraints,
                success_metrics,
                action_history,
                context_embedding
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            game_context.game_name,
            game_context.action_type.value,
            game_context.current_state,
            Json([]),  # Initial core concepts (empty until analyzed)
            "",  # Initial action pattern (empty until analyzed)
            Json(game_context.constraints),
            Json({"success_rate": 0.0}),  # Initial success metrics
            Json(action_history_json),
            embedding
            ))
            self.conn.commit()

        print('Completed ingestion of', game_context.game_name)

    async def find_similar_strategies(self, current_state: str, n_similar: int) -> List[Strategy]:
        # 1. Get embedding for current state using OpenAI
        embedding = await self._generate_embedding(current_state)
        
        # 2. Query database for similar strategies using vector similarity
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    game_name,
                    action_type,
                    context,
                    core_concepts,
                    action_pattern,
                    constraints,
                    success_metrics,
                    action_history
                FROM game_strategies
                WHERE context_embedding IS NOT NULL
                ORDER BY context_embedding <=> %s::vector
                LIMIT %s
            """, (embedding, n_similar))
            
            results = cur.fetchall()
        
        # 3. Convert results to Strategy objects
        strategies = []
        for result in results:
            action_history = []
            if result[7]:  # action_history JSON
                for action in result[7]:
                    action_history.append(
                        GameAction(
                            action_name=action['action_name'],
                            state_before=action['state_before'],
                            state_after=action['state_after'],
                            result=action['result']
                        )
                    )
            
            context = GameContext(
                game_name=result[0],
                current_state=result[2],
                action_type=ActionType(result[1]),
                constraints=result[5],
                available_actions=[],  # Would need to be stored in DB or derived
                action_history=action_history
            )
            
            strategy = Strategy(
                context=context,
                core_concepts=result[3],
                action_pattern=result[4],
                expected_outcome="",  # Not stored directly in DB
                success_rate=result[6].get('success_rate', 0.0) if result[6] else 0.0,
                action_history=action_history
            )
            
            strategies.append(strategy)
        
        return strategies

# --- Strategy Translation Engine ---

class StrategyTranslator:
    def __init__(self):
        load_dotenv()
        # Remove unused client instances - we'll create them when needed
        
        # Core concept mappings across different game types
        self.concept_mappings = {
            "positional": {
                "control": ["area_control", "position_strength", "mobility"],
                "tempo": ["timing", "initiative", "pressure"],
                "material": ["resources", "pieces", "assets"]
            },
            "text_based": {
                "pattern_recognition": ["word_patterns", "text_structure", "syntax"],
                "resource_management": ["word_length", "letter_frequency", "vocabulary"]
            },
            "logical": {
                "constraint_satisfaction": ["rule_application", "deduction", "inference"],
                "elimination": ["pruning", "reduction", "simplification"]
            },
            "persuasion": {
                "influence": ["negotiation", "convincing", "leverage"],
                "information_management": ["hidden_info", "bluffing", "revelation"]
            },
            "resource_management": {
                "economy": ["resource_acquisition", "efficiency", "stockpiling"],
                "conversion": ["resource_transformation", "value_creation", "synergy"],
                "timing": ["investment", "spending_cycles", "opportunity_cost"],
                "scarcity": ["prioritization", "rationing", "substitution"]
            }
        }

    async def translate_strategy(
        self,
        source_strategy: Strategy,
        target_context: GameContext
    ) -> Strategy:
        """Translate a strategy from one game context to another"""
        
        # 1. Extract core strategic concepts
        prompt = self._build_concept_extraction_prompt(source_strategy)
        core_concepts = await self._get_core_concepts(prompt)

        # 2. Map concepts to target game type
        mapped_concepts = self._map_concepts_to_target(
            core_concepts,
            source_strategy.context.action_type,
            target_context.action_type
        )

        # 3. Generate adapted strategy
        adapted_strategy = await self._generate_adapted_strategy(
            mapped_concepts,
            target_context
        )

        return adapted_strategy

    def _build_concept_extraction_prompt(self, strategy: Strategy) -> str:
        return f"""
        Analyze this game strategy and identify the core strategic concepts:
        
        Game: {strategy.context.game_name}
        Current State: {strategy.context.current_state}
        Action Pattern: {strategy.action_pattern}
        Expected Outcome: {strategy.expected_outcome}
        
        Extract the fundamental strategic principles that could apply to other games.
        Focus on concepts like:
        - Resource management
        - Positional advantage
        - Information control
        - Timing/Initiative
        - Risk/Reward balance
        
        Return your answer as a JSON array of strings representing the core concepts.
        """

    async def _get_core_concepts(self, prompt: str) -> List[str]:
        try:
            # Create client for each request
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are a game strategy analyst. Extract core strategic concepts and return them as a JSON array of strings."
            )
            
            # Extract JSON from the response
            content = response.content[0].text
            # Find JSON array in the response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                try:
                    concepts = json.loads(json_match.group(0))
                    return concepts
                except json.JSONDecodeError:
                    print(f"Error parsing JSON from response: {content}")
                    return []
            else:
                # Fallback if no JSON array is found
                print(f"No JSON array found in response: {content}")
                return []
        except Exception as e:
            print(f"Error getting core concepts: {e}")
            return []

    def _map_concepts_to_target(
        self,
        concepts: List[str],
        source_type: ActionType,
        target_type: ActionType
    ) -> List[str]:
        mapped_concepts = []
        
        # Check if source_type and target_type exist in concept_mappings
        if source_type.value not in self.concept_mappings:
            print(f"Warning: Source type '{source_type.value}' not found in concept mappings")
            return concepts  # Return original concepts if source type not found
            
        if target_type.value not in self.concept_mappings:
            print(f"Warning: Target type '{target_type.value}' not found in concept mappings")
            return concepts  # Return original concepts if target type not found
        
        for concept in concepts:
            # Find equivalent concepts in target game type
            if concept in self.concept_mappings[source_type.value]:
                mapped = self.concept_mappings[target_type.value].get(
                    concept,
                    self.concept_mappings[target_type.value].get("general", [])
                )
                mapped_concepts.extend(mapped)
            else:
                # If concept not found in mappings, keep the original concept
                mapped_concepts.append(concept)
        
        return list(set(mapped_concepts))  # Remove duplicates

    async def _generate_adapted_strategy(
        self,
        mapped_concepts: List[str],
        target_context: GameContext
    ) -> Strategy:
        prompt = f"""
        Generate a specific strategy for this game context using these strategic concepts:
        
        Game: {target_context.game_name}
        Current State: {target_context.current_state}
        Available Actions: {', '.join(target_context.available_actions)}
        Constraints: {', '.join(target_context.constraints)}
        
        Strategic Concepts to Apply: {', '.join(mapped_concepts)}
        
        Provide a JSON object with these fields:
        1. action_pattern: A specific pattern of actions to follow
        2. expected_outcome: What the player can expect to achieve
        3. estimated_success_rate: A number between 0 and 1 representing probability of success
        """

        try:
            # Create client for each request
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are a game strategy advisor. Generate a strategy and return it as a JSON object."
            )
            
            # Extract JSON from the response
            content = response.content[0].text
            # Find JSON object in the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    strategy_data = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    print(f"Error parsing JSON from response: {content}")
                    strategy_data = {
                        "action_pattern": "",
                        "expected_outcome": "",
                        "estimated_success_rate": 0.0
                    }
            else:
                # Fallback if no JSON object is found
                print(f"No JSON object found in response: {content}")
                strategy_data = {
                    "action_pattern": "",
                    "expected_outcome": "",
                    "estimated_success_rate": 0.0
                }
        except Exception as e:
            print(f"Error generating adapted strategy: {e}")
            strategy_data = {
                "action_pattern": "",
                "expected_outcome": "",
                "estimated_success_rate": 0.0
            }
        
        return Strategy(
            context=target_context,
            core_concepts=mapped_concepts,
            action_pattern=strategy_data.get("action_pattern", ""),
            expected_outcome=strategy_data.get("expected_outcome", ""),
            success_rate=strategy_data.get("estimated_success_rate", 0.0)
        )

# --- Main Game Strategy Pipeline ---

class GameStrategyPipeline:
    def __init__(self):
        load_dotenv()
        self.vector_store = VectorStore()
        # Initialize database tables
        self.vector_store.setup_database()
        self.translator = StrategyTranslator()
        # Remove unused client instances - we'll create them when needed
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    async def process_new_game(
        self,
        game_context: GameContext,
        n_similar_strategies: int = 3
    ) -> Strategy:

        # 1. ingest into vector database first
        await self.vector_store._ingest_new_game(game_context)

        # 2. Find similar strategies from known games
        similar_strategies = await self.vector_store.find_similar_strategies(
            game_context.current_state,
            n_similar_strategies
        )

        # 3. Translate each strategy to new game context
        # this is more conceptual / abstract: we just consider the game's ruleset and description, and not the LLM's action_history
        adapted_strategies = []
        for strategy in similar_strategies:
            adapted = await self.translator.translate_strategy(
                strategy,
                game_context
            )
            adapted_strategies.append(adapted)

        # 4. Synthesize strategies
        # this is more empirical, where we try to get the LLM to learn from its action_history as well
        final_strategy = await self._synthesize_strategies(
            adapted_strategies,
            game_context
        )

        return final_strategy

    async def _synthesize_strategies(
        self,
        strategies: List[Strategy],
        target_context: GameContext
    ) -> Strategy:
        # extract past actions
        action_histories = []

        for strategy in strategies:
            if strategy.action_history:
                action_histories.extend([
                    {
                        'action': action.action_name,
                        'before': action.state_before,
                        'after': action.state_after,
                        'result': action.result
                    }
                    for action in strategy.action_history
                ])

        # Combine insights from multiple strategies
        prompt = f"""
        Synthesize these adapted strategies into a single optimal strategy:
        
        Game Context:
        {target_context.current_state}
        
        Available Actions:
        {', '.join(target_context.available_actions)}

        Previous Action History:
        {json.dumps(action_histories, indent=2)}
        
        Adapted Strategies:
        {json.dumps([{"core_concepts": s.core_concepts, "action_pattern": s.action_pattern, "expected_outcome": s.expected_outcome} for s in strategies], indent=2)}
        
        Provide a JSON object with these fields:
        1. core_concepts: Array of core strategic concepts
        2. action_pattern: A specific pattern of actions to follow
        3. expected_outcome: What the player can expect to achieve
        4. estimated_success_rate: A number between 0 and 1 representing probability of success
        """

        try:
            # Create client for each request
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are a game strategy synthesizer. Combine multiple strategies into an optimal one and return it as a JSON object."
            )
            
            # Extract JSON from the response
            content = response.content[0].text
            # Find JSON object in the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    synthesis = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    print(f"Error parsing JSON from response: {content}")
                    synthesis = {
                        "core_concepts": [],
                        "action_pattern": "",
                        "expected_outcome": "",
                        "estimated_success_rate": 0.0
                    }
            else:
                # Fallback if no JSON object is found
                print(f"No JSON object found in response: {content}")
                synthesis = {
                    "core_concepts": [],
                    "action_pattern": "",
                    "expected_outcome": "",
                    "estimated_success_rate": 0.0
                }
        except Exception as e:
            print(f"Error synthesizing strategies: {e}")
            synthesis = {
                "core_concepts": [],
                "action_pattern": "",
                "expected_outcome": "",
                "estimated_success_rate": 0.0
            }
        
        return Strategy(
            context=target_context,
            core_concepts=synthesis.get("core_concepts", []),
            action_pattern=synthesis.get("action_pattern", ""),
            expected_outcome=synthesis.get("expected_outcome", ""),
            success_rate=synthesis.get("estimated_success_rate", 0.0),
            action_history=target_context.action_history
        )

# --- Usage Example ---

async def main():
    # Initialize pipeline
    pipeline = GameStrategyPipeline()

    try:
        # Example game context
        context = GameContext(
            game_name="Unknown Game X",
            current_state="Players start with 5 cards each. Board has 3 resource spots.",
            action_type=ActionType.RESOURCE_MANAGEMENT,
            constraints=["Max 2 cards per turn", "Cannot repeat same action twice"],
            available_actions=["Play Card", "Collect Resource", "Trade"]
        )

        # Process new game
        strategy = await pipeline.process_new_game(context)
        
        # Print results
        print(f"\nFinal Strategy:")
        print(f"Core Concepts: {strategy.core_concepts}")
        print(f"Action Pattern: {strategy.action_pattern}")
        print(f"Expected Outcome: {strategy.expected_outcome}")
        print(f"Estimated Success Rate: {strategy.success_rate}")
    finally:
        # Clean up resources
        if pipeline.vector_store.conn:
            pipeline.vector_store.close()

if __name__ == "__main__":
    asyncio.run(main())