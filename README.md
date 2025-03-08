# considerations

1. analyze environment state (memory)
2. consider list of tools available for model to call
3. consider strategy that we should adopt given the state
4. consider type of game that we're dealing with

primarily will need flexible state management as the rulset for each game differs greatly

RAG pipeline for the ingestion of games + rulesets as groundtruth

# components

1. game state parser / interpreter (we might need a way to store memory if the game state drags out)
2. decision-making pipeline (action validator / executor + strategy optimizer (learns from previous moves / games))
3. MCP integration layer
4. Response formatter
5. performance monitoring / testing (compare against model answers?)

## memory

https://github.com/smithery-ai/reference-servers/tree/main/src/memory

- short term memory: recent actions / immediate context
- working memory: current game state / active goals (past strategies / games and their success/failure)
- long term memory: learned strategies and patterns

## RAG pipeline

- figure out embedding model type to use for ingesting games + rulesets
- postgres + pgvector to store exact matches + semantic searches (?)
- what similarity metric to use to decide how similar games might be (so that the model can use for retrieval)
- what indexing method to use for performance

# Hackathon rules

have to compete @ least 10x PER environment / game to qualify in public + private set categories

3 hidden games in private dataset at 530pm

180 secs timeout per turn b4 model gets kicked

## tool use

- get it to write code to reason through the game requirements
- access the .reason key in the return info object for each turn of game, to understand where it could do better
-
