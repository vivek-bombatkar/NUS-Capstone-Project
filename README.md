# Multi-Agent AI Assistant with RAG, LLM Routing & Image Generation

## Overview

This project implements a fully integrated multi-agent AI system that combines conversational AI, document intelligence, structured data querying, and generative capabilities into a single application. The system is designed to simulate a real-world AI assistant that can process user queries, route them intelligently, and coordinate multiple specialized agents to generate meaningful outputs.

---

## Business Use Case: AI based customer feedback management system

This system  aims to help small and medium-sized enterprises (SMEs) transform large volumes of customer feedback into actionable business insights. Organizations often store feedback across documents, databases, and reports but lack a unified way to extract value from it. This solution demonstrates how AI can centralize feedback analysis, identify sentiment and trends, answer questions from internal documents, and even generate visual representations, enabling better decision-making and faster response to customer needs.

---

## Objectives Achieved

- Conversational interface with limited memory
- Document-based Question Answering using RAG
- Text-to-image generation with prompt engineering
- Multi-agent orchestration through a controller
- LLM-based routing and reasoning
- Fully integrated end-to-end system

---

## Technical Architecture

### High-Level Flow

User Input → Streamlit UI → Controller (LLM Router) → Multi-Agent System → Formatter → UI Output

---

### Core Architectural Layers

#### 1. Presentation Layer (Streamlit)

- Chat-based UI for user interaction
- Maintains session-level conversational memory
- Displays structured text and generated images
- Acts as the entry point for all user queries

---

#### 2. Controller Layer (Router)

- Implements LLM-based routing instead of rule-based logic
- Uses Groq LLM to classify user intent dynamically
- Outputs a list of agents to be executed
- Supports multi-agent invocation for complex queries

Example:
- Query: "Why are customers unhappy?" → Sentiment + Trend
- Query: "What should I improve?" → Recommendation
- Query: "Summarize reports" → RAG

---

#### 3. Multi-Agent Layer

Each agent is modular and independently executable but coordinated through the controller.

##### Sentiment Agent
- Analyzes feedback stored in SQLite database
- Computes sentiment distribution (positive, negative, neutral)

##### Trend Agent
- Identifies recurring issues using keyword aggregation
- Detects patterns such as delivery delays or product defects

##### Recommendation Agent (LLM-based)
- Consumes outputs from sentiment and trend agents
- Uses LLM reasoning to generate actionable business recommendations

##### SQL Agent
- Executes structured queries on SQLite database
- Provides numerical insights such as counts and aggregates

##### RAG Agent (LLM-based Retrieval)
- Uses internal document store as knowledge base
- LLM selects relevant documents dynamically
- Generates grounded answers based on retrieved content

##### Image Agent
- Converts user query into optimized prompt using LLM
- Calls Hugging Face Inference API for image generation
- Returns image path for UI rendering

---

#### 4. Data Layer

- SQLite database for structured feedback storage
- Static document repository for RAG
- Local file storage for generated images

---

#### 5. Formatter Layer

- Aggregates outputs from multiple agents
- Structures response into sections:
  - Sentiment Analysis
  - Trends
  - Recommendations
  - Data Insights
  - Knowledge Insights
  - Visualization

---

## Multi-Agent Coordination

Agents are orchestrated in a dependency-aware pipeline:

Sentiment → Trend → Recommendation

- Recommendation agent uses outputs from prior agents
- Enables context-aware reasoning instead of isolated execution

---

## LLM Integration

The system uses Groq-hosted LLaMA models for:

- Query classification (routing)
- Recommendation generation
- Prompt engineering for image generation
- RAG retrieval and answer generation

---

## RAG Design

### Constraint

Traditional RAG approaches using FAISS, ChromaDB, or sentence-transformers were not feasible due to platform limitations on macOS Intel.

### Solution

Implemented LLM-based retrieval:

Query → LLM selects relevant documents → LLM generates answer

This approach ensures:
- No dependency on external vector databases
- Simpler architecture
- Valid document-grounded responses

---

## Image Generation Design

- LLM used to transform user query into detailed prompt
- Hugging Face Inference API used for image generation
- Generated image saved locally and rendered in UI

---

## Challenges and Solutions

### Dependency Issues
- FAISS, Torch, and ChromaDB incompatibility
- Resolved by using LLM-based RAG

### API Instability
- Hugging Face model deprecations and permission issues
- Resolved by switching to supported models and adding fallbacks

### Encoding Errors
- Unicode characters from LLM caused API failures
- Resolved using ASCII sanitization

### Agent Coordination
- Initial design lacked dependency flow
- Introduced structured orchestration pipeline

---

## Debugging Approach

- Isolated testing of each agent
- Incremental integration
- Logging API responses
- Handling failures with fallback mechanisms

---

## Improvements Implemented

- Replaced rule-based routing with LLM-based routing
- Introduced context sharing across agents
- Improved output formatting for clarity
- Added prompt engineering for better generation quality
- Implemented robust error handling

---

## Future Enhancements

- Integrate vector databases (FAISS, Pinecone)
- Improve RAG with document chunking
- Add authentication and access control
- Deploy on cloud platforms (e.g., Hugging Face Spaces)
- Introduce feedback learning loop

---

## Example Use Cases

### Customer Insight Analysis
Input: Why are customers unhappy?
Output: Sentiment + Trends + Insights

### Decision Support
Input: What should I improve?
Output: Actionable recommendations

### Document QA
Input: Summarize issues from reports
Output: RAG-based grounded response

### Visualization
Input: Generate dashboard image
Output: Prompt-engineered image

---

## Conclusion

This project demonstrates how multiple AI capabilities can be integrated into a cohesive system. It highlights the use of LLMs not just for generation, but for orchestration, reasoning, and decision-making. The system reflects real-world AI architecture patterns and shows how constraints can be addressed through adaptive design choices.

---

## Final Status

- Multi-agent system implemented
- LLM routing enabled
- RAG pipeline functional
- SQL querying integrated
- Image generation working
- UI fully integrated

This represents a production-style prototype for a GenAI-powered business assistant.
