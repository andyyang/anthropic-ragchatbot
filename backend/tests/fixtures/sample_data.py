"""
Sample data fixtures for testing the RAG system components.
Contains realistic course content and expected API responses.
"""

from typing import Dict, List, Any

# Sample Anthropic API responses for different scenarios
ANTHROPIC_RESPONSES = {
    "direct_response": {
        "content": [{"type": "text", "text": "This is a direct response without tool use."}],
        "stop_reason": "end_turn"
    },
    
    "tool_use_response": {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "search_course_content",
                "input": {
                    "query": "MCP basics",
                    "course_name": "Introduction to MCP"
                }
            }
        ],
        "stop_reason": "tool_use"
    },
    
    "final_tool_response": {
        "content": [{"type": "text", "text": "Based on the search results, MCP (Model Context Protocol) is a revolutionary approach to AI integration that enables seamless communication between AI models and external tools."}],
        "stop_reason": "end_turn"
    },
    
    "outline_tool_response": {
        "content": [
            {
                "type": "tool_use", 
                "id": "toolu_456",
                "name": "get_course_outline",
                "input": {
                    "course_title": "Introduction to MCP"
                }
            }
        ],
        "stop_reason": "tool_use"
    }
}

# Sample course content for different test scenarios
SAMPLE_COURSE_CONTENT = {
    "mcp_lesson_1": """
    # Getting Started with MCP

    MCP (Model Context Protocol) is a revolutionary approach to AI integration that enables 
    seamless communication between AI models and external tools and data sources.

    ## Key Concepts:
    - Protocol-based communication
    - Tool integration
    - Context management
    - Real-time data access

    In this lesson, you'll learn the fundamentals of MCP and how to get started with basic implementations.
    """,
    
    "mcp_lesson_2": """
    # Basic Concepts of MCP

    This lesson covers the core concepts that make MCP powerful:

    ## Architecture
    MCP uses a client-server architecture where:
    - Clients (AI applications) connect to servers
    - Servers provide tools and resources
    - Communication happens via JSON-RPC

    ## Protocol Features
    - Standardized tool calling
    - Resource management
    - Progress tracking
    - Error handling
    """,
    
    "python_decorators": """
    # Python Decorators and Generators

    Decorators are a powerful Python feature that allows you to modify or extend 
    the behavior of functions or classes without permanently modifying their code.

    ## Basic Decorator Syntax:
    ```python
    @decorator_function
    def my_function():
        pass
    ```

    ## Common Use Cases:
    - Logging
    - Authentication
    - Caching
    - Rate limiting
    """
}

# Expected search results for different queries
EXPECTED_SEARCH_RESULTS = {
    "mcp_basics": {
        "documents": [
            "MCP (Model Context Protocol) is a revolutionary approach to AI integration that enables seamless communication between AI models and external tools and data sources.",
            "This lesson covers the core concepts that make MCP powerful: Architecture, Protocol Features"
        ],
        "metadata": [
            {"course_title": "Introduction to MCP", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Introduction to MCP", "lesson_number": 2, "chunk_index": 1}
        ],
        "distances": [0.1, 0.15]
    },
    
    "python_decorators": {
        "documents": [
            "Decorators are a powerful Python feature that allows you to modify or extend the behavior of functions or classes without permanently modifying their code."
        ],
        "metadata": [
            {"course_title": "Advanced Python Programming", "lesson_number": 1, "chunk_index": 0}
        ],
        "distances": [0.12]
    },
    
    "empty_results": {
        "documents": [],
        "metadata": [],
        "distances": []
    }
}

# Course outline responses  
COURSE_OUTLINE_RESPONSES = {
    "mcp_course": """**Introduction to MCP**
*Instructor: Dr. Smith*
*Course Link: https://example.com/mcp*

**Course Outline:**
Lesson 1: Getting Started
Lesson 2: Basic Concepts  
Lesson 3: Advanced Topics
""",
    
    "python_course": """**Advanced Python Programming**
*Instructor: Dr. Johnson*
*Course Link: https://example.com/python*

**Course Outline:**
Lesson 1: Decorators and Generators
Lesson 2: Async Programming
"""
}

# Error scenarios for testing error handling
ERROR_SCENARIOS = {
    "vector_store_connection_error": "Failed to connect to vector database",
    "anthropic_api_error": "Anthropic API rate limit exceeded", 
    "invalid_course_name": "No course found matching 'NonExistent Course'",
    "tool_execution_error": "Tool execution failed: invalid parameters"
}

# Test query scenarios with expected behavior
TEST_QUERIES = {
    "content_search": {
        "query": "What is MCP and how does it work?",
        "expected_tool": "search_course_content",
        "expected_params": {"query": "What is MCP and how does it work?"}
    },
    
    "course_specific": {
        "query": "Tell me about decorators in Python programming",
        "expected_tool": "search_course_content", 
        "expected_params": {
            "query": "Tell me about decorators in Python programming",
            "course_name": "Advanced Python Programming"
        }
    },
    
    "course_outline": {
        "query": "What lessons are covered in the MCP course?",
        "expected_tool": "get_course_outline",
        "expected_params": {"course_title": "Introduction to MCP"}
    },
    
    "general_knowledge": {
        "query": "What is the capital of France?",
        "expected_tool": None,  # Should not use tools
        "direct_response": "The capital of France is Paris."
    }
}

# Mock ChromaDB collection responses
CHROMA_RESPONSES = {
    "course_search": {
        "documents": [["MCP content", "Python content"]], 
        "metadatas": [[
            {"course_title": "Introduction to MCP", "lesson_number": 1},
            {"course_title": "Advanced Python Programming", "lesson_number": 1}
        ]],
        "distances": [[0.1, 0.2]]
    },
    
    "course_metadata": {
        "metadatas": [{
            "title": "Introduction to MCP",
            "instructor": "Dr. Smith",
            "course_link": "https://example.com/mcp",
            "lessons_json": '[{"lesson_number": 1, "lesson_title": "Getting Started", "lesson_link": "https://example.com/mcp/lesson1"}]',
            "lesson_count": 3
        }]
    }
}