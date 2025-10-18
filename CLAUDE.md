# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
- **Start development server**: `./run.sh` or `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Install dependencies**: `uv sync`
- **Create docs directory**: `mkdir -p docs` (required for startup)

### Environment Setup
- Create `.env` file in root with: `ANTHROPIC_API_KEY=your_key_here`
- Requires Python 3.13+, uv package manager, and Anthropic API key

## Architecture

This is a **Retrieval-Augmented Generation (RAG) system** for answering questions about course materials using semantic search and AI.

### Core Components (all in `backend/`)

**RAGSystem** (`rag_system.py`) - Main orchestrator that coordinates all components
- Manages document processing pipeline
- Handles user queries with conversation context
- Integrates vector search with AI generation using Claude Sonnet 4

**VectorStore** (`vector_store.py`) - ChromaDB-based semantic search
- Two collections: `course_catalog` (metadata) and `course_content` (chunks)
- Uses sentence-transformers for embeddings (all-MiniLM-L6-v2)
- Supports filtering by course name and lesson number

**DocumentProcessor** (`document_processor.py`) - Converts documents to structured data
- Processes PDF, DOCX, TXT files into Course/Lesson objects
- Creates searchable chunks (800 chars, 100 overlap)

**AIGenerator** (`ai_generator.py`) - Claude integration with tool support
- Uses function calling for semantic search operations
- Maintains conversation history for context

**SessionManager** (`session_manager.py`) - Conversation state management
- Tracks user sessions with limited history (2 exchanges)

### Data Flow
1. Documents in `docs/` → DocumentProcessor → Course/Lesson objects
2. Course metadata → `course_catalog` collection
3. Content chunks → `course_content` collection  
4. User query → AI with search tools → VectorStore → Contextual response

### Key Models (`models.py`)
- `Course`: Contains title, instructor, lessons
- `Lesson`: Individual lesson with number, title, link
- `CourseChunk`: Searchable content unit with metadata

### FastAPI App (`app.py`)
- `/api/query` - Main RAG endpoint
- `/api/courses` - Course analytics
- Serves static frontend from `../frontend`
- Auto-loads documents from `docs/` on startup

### Configuration (`config.py`)
- Environment-based settings via `.env`
- ChromaDB path: `./chroma_db`
- Chunk settings: 800 size, 100 overlap, 5 max results
- always use uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies
- don't run the server using ./run.sh I will start it myself