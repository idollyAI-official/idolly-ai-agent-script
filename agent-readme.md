# AI Agent Readme (ai-agent-readme.md)

## Overview

**Idolly** is a **social and influencer AI agent platform** for generating AI images and videos, as well as minting NFTs on Web3.  
This document highlights the agent server architecture and core components (the agent, prompt enhancement module, prompt calculator, Flowise AI instance, etc.) in relation to our submission to the Solana AI hackathon’s [social and influencer agents] track.

---

## Key Components

### 1. Prompt Enhancer

- **Overview**  
  The Prompt Enhancer analyzes user inputs (prompts) and automatically **expands or refines** them into a **prompt template** optimized for image (or video) generation.  
  Internally, it uses an LLM chain to dynamically complete and piece together a `Prompt Template`.

- **How It Works**  
  1. **User Input**: e.g., “anime style character, futuristic cityscape”  
  2. **LLM Chain** uses something like a `PromptEnhancerTemplate` (e.g., `"{user_input} in style {style}, lighting={lighting}, ..."`) to generate a finalized prompt.  
  3. **Result**: “High-detail anime character in a futuristic cityscape, style=cyberpunk, …”  
  4. This refined prompt is sent to the **image engine** (ComfyUI etc.) to generate and return the final output.

- **Code Snippet**

  ```python
  from langchain import LLMChain
  from langchain.prompts import PromptTemplate
  from .prompt_templates import PROMPT_ENHANCER_TEMPLATE

  class PromptEnhancer:
      def __init__(self, llm):
          self.llm_chain = LLMChain(
              llm=llm,
              prompt=PromptTemplate(PROMPT_ENHANCER_TEMPLATE)
          )

      def enhance_prompt(self, user_input: str, style: str = "anime", resolution: str = "4K"):
          """
          Uses an LLM to combine user input, style etc.
          into the optimal image prompt.
          """
          return self.llm_chain.run({
              "user_input": user_input,
              "style": style,
              "model": model, ...
          })
  ```

---

### 2. Agent (RAG & Automated Agents)

- **Overview**  
  The core agent in Idolly runs on a RAG structure, powered by a Vector DB.
  - **Flowise AI** instances enable a **no-code, flow-based** approach to easily build and execute agent flows.  
  - Aims to offer **social and influencer** agents that respond in text, audio, or video.  
  - Examples include a **Video Script Generator**, **Prompt Assistant**, **LLM-based Retriever**, and more specialized sub-agents.

- **How It Works**  
  1. **Vector DB**: FAQ, script examples, documentation, marketing data, and so on are embedded and stored.  
  2. **LangChain RAG Pipeline**:  
     - User input → **Vector DB** searches for relevant documents/context → LLM uses this info to produce an optimal answer/action.  
  3. **Autonomous Agent**:  
     - Under certain conditions, it can generate follow-up prompts on its own or make subsequent API calls (e.g., NFT minting, text-to-video).

- **Sample Code Snippet**

  ```python
  from langchain.vectorstores import Chroma
  from langchain.embeddings import OpenAIEmbeddings
  from langchain.chains import RetrievalQA
  from langchain.chat_models import ChatOpenAI

  class AdvancedAgent:
      def __init__(self):
          self.embedding_fn = OpenAIEmbeddings()
          self.vstore = Chroma(embedding_function=self.embedding_fn, collection_name="idolly_docs")
          self.retriever = self.vstore.as_retriever()
          self.llm = ChatOpenAI(temperature=0.7)
          self.qa_chain = RetrievalQA.from_chain_type(
              llm=self.llm,
              retriever=self.retriever,
              chain_type="stuff"
          )

      def get_answer(self, query: str) -> str:
          """
          Performs RAG-based Q&A using the Vector DB and returns the answer.
          """
          return self.qa_chain.run(query)

      def run_autonomous_task(self, task: str):
          """
          Additional workflows can be triggered here, e.g., image generation, NFT minting, etc.
          """
          # Example mock logic
          if "mint_nft" in task:
              # NFT minting API call
              return "Initiating NFT minting workflow..."
          else:
              return "No recognized autonomous task."
  ```

- **Flowise AI Integration**  
  - Within **Flowise AI**, you can visually configure LangChain nodes/components (LLM, Vector DB, Agent, etc.).  
  - The Idolly server can call the Flowise AI instance via REST API, or Flowise can directly call our server via an HTTP request node.  
  - Simplified Example:


---

### 3. Prompt Calculator

- **Overview**  
  The **Prompt Calculator** disassembles user input into **multiple prompt candidates** (or queries), computes an **“effectiveness score”** (rarity, creative value, etc.) for each candidate, and **ranks** them.  
  It also helps set **prices** based on NFT rarity scores when minting.

- **How It Works**  
  1. **User Input**: "cyberpunk cat, neon background, epic lighting..."  
  2. **Split Queries**: "cyberpunk cat" / "neon background" / "epic lighting"  
  3. **Scoring**: Predictive image quality scores, rarity values etc.  
  4. **Result Ranking**
  5. **NFT Pricing**: Higher rarity yields slightly higher cost.