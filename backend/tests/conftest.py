import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any, Optional
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add the backend directory to the Python path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from config import Config

@pytest.fixture
def test_config():
    """Test configuration with safe paths and test settings"""
    config = Config()
    config.CHROMA_PATH = "./test_chroma_db"
    config.ANTHROPIC_API_KEY = "test_key"
    config.MAX_RESULTS = 3
    config.CHUNK_SIZE = 200
    config.CHUNK_OVERLAP = 50
    return config

@pytest.fixture
def sample_courses():
    """Sample course data for testing"""
    courses = [
        Course(
            title="Introduction to MCP",
            instructor="Dr. Smith", 
            course_link="https://example.com/mcp",
            lessons=[
                Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/mcp/lesson1"),
                Lesson(lesson_number=2, title="Basic Concepts", lesson_link="https://example.com/mcp/lesson2"),
                Lesson(lesson_number=3, title="Advanced Topics", lesson_link="https://example.com/mcp/lesson3")
            ]
        ),
        Course(
            title="Advanced Python Programming",
            instructor="Dr. Johnson",
            course_link="https://example.com/python", 
            lessons=[
                Lesson(lesson_number=1, title="Decorators and Generators", lesson_link="https://example.com/python/lesson1"),
                Lesson(lesson_number=2, title="Async Programming", lesson_link="https://example.com/python/lesson2")
            ]
        )
    ]
    return courses

@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing"""
    chunks = [
        CourseChunk(
            content="MCP (Model Context Protocol) is a revolutionary approach to AI integration.",
            course_title="Introduction to MCP",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="In this lesson, we'll explore the basic concepts of MCP and how it works.",
            course_title="Introduction to MCP", 
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Python decorators are a powerful feature that allows you to modify functions.",
            course_title="Advanced Python Programming",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Async programming in Python allows for concurrent execution of code.",
            course_title="Advanced Python Programming",
            lesson_number=2,
            chunk_index=1
        )
    ]
    return chunks

@pytest.fixture
def mock_search_results():
    """Mock search results for different test scenarios"""
    def _create_results(scenario="success"):
        if scenario == "success":
            return SearchResults(
                documents=[
                    "MCP (Model Context Protocol) is a revolutionary approach to AI integration.",
                    "In this lesson, we'll explore the basic concepts of MCP and how it works."
                ],
                metadata=[
                    {"course_title": "Introduction to MCP", "lesson_number": 1},
                    {"course_title": "Introduction to MCP", "lesson_number": 2}
                ],
                distances=[0.1, 0.2]
            )
        elif scenario == "empty":
            return SearchResults(
                documents=[],
                metadata=[],
                distances=[]
            )
        elif scenario == "error":
            return SearchResults.empty("Vector store connection failed")
        elif scenario == "course_filtered":
            return SearchResults(
                documents=[
                    "Python decorators are a powerful feature that allows you to modify functions."
                ],
                metadata=[
                    {"course_title": "Advanced Python Programming", "lesson_number": 1}
                ],
                distances=[0.15]
            )
    return _create_results

@pytest.fixture
def mock_vector_store(mock_search_results):
    """Mock VectorStore with controllable behavior"""
    mock_store = Mock()
    
    # Default successful search
    mock_store.search.return_value = mock_search_results("success")
    
    # Mock course resolution
    mock_store._resolve_course_name.return_value = "Introduction to MCP"
    
    # Mock lesson link retrieval
    mock_store.get_lesson_link.return_value = "https://example.com/mcp/lesson1"
    
    # Mock course catalog operations
    mock_store.course_catalog.get.return_value = {
        'metadatas': [{
            'title': 'Introduction to MCP',
            'instructor': 'Dr. Smith',
            'course_link': 'https://example.com/mcp',
            'lessons_json': '[{"lesson_number": 1, "lesson_title": "Getting Started", "lesson_link": "https://example.com/mcp/lesson1"}]'
        }]
    }
    
    return mock_store

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing AI responses"""
    mock_client = Mock()
    
    # Mock successful response without tool use
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a direct response from Claude.")]
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    
    return mock_client

@pytest.fixture
def mock_anthropic_tool_response():
    """Mock Anthropic response that includes tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"
    
    # Mock tool use content block
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "MCP basics", "course_name": "Introduction to MCP"}
    tool_block.id = "tool_call_123"
    
    mock_response.content = [tool_block]
    
    # Mock final response after tool execution
    final_response = Mock()
    final_response.content = [Mock(text="Based on the course content, MCP is a revolutionary approach to AI integration.")]
    
    return mock_response, final_response

@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager for testing"""
    mock_manager = Mock()
    
    # Mock tool definitions
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "course_name": {"type": "string", "description": "Course name"}
                },
                "required": ["query"]
            }
        }
    ]
    
    # Mock tool execution
    mock_manager.execute_tool.return_value = "MCP content search results"
    
    # Mock source tracking
    mock_manager.get_last_sources.return_value = [
        {"text": "Introduction to MCP - Lesson 1", "url": "https://example.com/mcp/lesson1"}
    ]
    
    return mock_manager

@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API testing"""
    mock_rag = Mock()
    
    # Mock query method
    mock_rag.query.return_value = (
        "This is a test response from the RAG system.",
        [{"text": "Test Source - Lesson 1", "url": "https://example.com/test/lesson1"}]
    )
    
    # Mock session manager
    mock_rag.session_manager.create_session.return_value = "test_session_123"
    mock_rag.session_manager.clear_session = Mock()
    
    # Mock analytics
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Introduction to MCP", "Advanced Python Programming"]
    }
    
    return mock_rag

@pytest.fixture
def test_app():
    """Create a test FastAPI application without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Pydantic models (copied from app.py)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class ClearSessionRequest(BaseModel):
        session_id: str

    class SourceItem(BaseModel):
        text: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Create test app
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # API Endpoints (using app.state.rag_system which will be set by fixture)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            rag_system = app.state.rag_system
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            
            answer, sources = rag_system.query(request.query, session_id)
            
            source_items = []
            for source in sources:
                if isinstance(source, dict):
                    source_items.append(SourceItem(text=source["text"], url=source.get("url")))
                else:
                    source_items.append(SourceItem(text=str(source), url=None))
            
            return QueryResponse(
                answer=answer,
                sources=source_items,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            rag_system = app.state.rag_system
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/clear-session")
    async def clear_session(request: ClearSessionRequest):
        try:
            rag_system = app.state.rag_system
            rag_system.session_manager.clear_session(request.session_id)
            return {"message": "Session cleared successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "RAG System API - Test Environment"}
    
    # Initialize with a placeholder - will be replaced by test fixture
    app.state.rag_system = None
    
    return app

@pytest.fixture
def test_client(test_app, mock_rag_system):
    """Create test client for API testing with mocked RAG system"""
    # Replace the mock RAG system in the app with our fixture
    test_app.state.rag_system = mock_rag_system
    return TestClient(test_app)

@pytest.fixture(autouse=True)
def cleanup_test_db():
    """Automatically cleanup test database after each test"""
    yield
    # Cleanup test database if it exists
    test_db_path = "./test_chroma_db"
    if os.path.exists(test_db_path):
        import shutil
        shutil.rmtree(test_db_path)