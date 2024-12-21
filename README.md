# blocksworld_rag_react
Adding ReAct and RAG + ReAct for generations on the Blocksworld environment for LLM generations

I added "t1_react" and "t1_rag_react" tasks to the existing Blocksworld repo. These tasks allowed for planning within the Blocksworld environment leveraging ReAct and ReAct combined with RAG.

ReAct: At every planning step, the model was reprompted with the added "observation" information that specificied the state of the environment at that time

RAG_ReAct: Here we used RAG to retrieve the relevant top_k (reasoning, action, observation) tuples from the past, so the context length is much shorter per generation. There were 3 different retrievers used for this BM25, TF-IDF, and the "all-MiniLM-L6-v2" sentence embedder model.

I tried a bunch of techniques to help improve generation quality including special end tokens at the end of each of the reasoning, action, and observation steps, provided more direct rules and context on the problem within the prompt, and included few shot examples of the ReAct generations. These various techniques didn't lead to any improvements over normally running the model on each question (for GPT-4 and GPT-4o), but this is most likely due to the model repeating its past step much more often with this way of prompt and diffculties parsing model generations each time its being prompted.
