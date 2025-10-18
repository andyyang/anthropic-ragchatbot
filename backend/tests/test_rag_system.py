"""
End-to-end tests for RAG system content-query handling.
Tests the complete pipeline from query to response with source tracking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
from pathlib import Path

# Add backend to path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk
from .fixtures.sample_data import TEST_QUERIES, ANTHROPIC_RESPONSES


class TestRAGSystemInitialization:
    """Test RAG system initialization and component setup"""
    
    def test_init_with_config(self, test_config):
        """Test RAG system initializes all components correctly"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):
            
            # Act
            rag = RAGSystem(test_config)
            
            # Assert
            assert rag.config == test_config
            assert hasattr(rag, 'vector_store')
            assert hasattr(rag, 'ai_generator')
            assert hasattr(rag, 'document_processor')
            assert hasattr(rag, 'session_manager')
            assert hasattr(rag, 'tool_manager')
            assert hasattr(rag, 'search_tool')
            assert hasattr(rag, 'outline_tool')
    
    def test_tools_registered_correctly(self, test_config):
        """Test that search and outline tools are registered with tool manager"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):
            
            # Act
            rag = RAGSystem(test_config)
            
            # Assert
            tool_definitions = rag.tool_manager.get_tool_definitions()
            tool_names = [tool['name'] for tool in tool_definitions]
            assert 'search_course_content' in tool_names
            assert 'get_course_outline' in tool_names


class TestRAGSystemContentQueries:
    """Test RAG system handling of content-query related questions"""
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator') 
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_content_query_with_tool_use(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test content query that triggers CourseSearchTool"""
        # Arrange
        rag = RAGSystem(test_config)
        
        # Mock AI response with tool use
        mock_ai_gen.return_value.generate_response.return_value = "MCP is a protocol for AI integration based on the course content."
        
        # Mock session manager
        mock_session.return_value.get_conversation_history.return_value = None
        mock_session.return_value.add_exchange = Mock()
        
        # Mock tool manager to return sources
        rag.tool_manager.get_last_sources = Mock(return_value=[
            {"text": "Introduction to MCP - Lesson 1", "url": "https://example.com/mcp/lesson1"}
        ])
        rag.tool_manager.reset_sources = Mock()
        
        # Act
        response, sources = rag.query("What is MCP and how does it work?")
        
        # Assert
        assert response == "MCP is a protocol for AI integration based on the course content."
        assert len(sources) == 1
        assert sources[0]["text"] == "Introduction to MCP - Lesson 1"
        
        # Verify AI generator was called with tools
        mock_ai_gen.return_value.generate_response.assert_called_once()
        call_args = mock_ai_gen.return_value.generate_response.call_args
        assert call_args[1]['tools'] == rag.tool_manager.get_tool_definitions()
        assert call_args[1]['tool_manager'] == rag.tool_manager
        
        # Verify sources were retrieved and reset
        rag.tool_manager.get_last_sources.assert_called_once()
        rag.tool_manager.reset_sources.assert_called_once()
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor') 
    @patch('rag_system.SessionManager')
    def test_course_specific_content_query(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test content query specific to a course"""
        # Arrange
        rag = RAGSystem(test_config)
        
        mock_ai_gen.return_value.generate_response.return_value = "Python decorators allow you to modify function behavior..."
        mock_session.return_value.get_conversation_history.return_value = None
        mock_session.return_value.add_exchange = Mock()
        
        rag.tool_manager.get_last_sources = Mock(return_value=[
            {"text": "Advanced Python Programming - Lesson 1", "url": "https://example.com/python/lesson1"}
        ])
        rag.tool_manager.reset_sources = Mock()
        
        # Act
        response, sources = rag.query("Tell me about decorators in Python programming")
        
        # Assert
        assert "decorators" in response.lower()
        assert len(sources) == 1
        assert "Python Programming" in sources[0]["text"]
        
        # Verify the query was processed with tools
        mock_ai_gen.return_value.generate_response.assert_called_once()
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_course_outline_query(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test query that should trigger course outline tool"""
        # Arrange
        rag = RAGSystem(test_config)
        
        mock_ai_gen.return_value.generate_response.return_value = """**Introduction to MCP**
*Instructor: Dr. Smith*

**Course Outline:**
Lesson 1: Getting Started
Lesson 2: Basic Concepts"""
        
        mock_session.return_value.get_conversation_history.return_value = None
        mock_session.return_value.add_exchange = Mock()
        
        rag.tool_manager.get_last_sources = Mock(return_value=[])
        rag.tool_manager.reset_sources = Mock()
        
        # Act
        response, sources = rag.query("What lessons are covered in the MCP course?")
        
        # Assert
        assert "Course Outline:" in response
        assert "Lesson 1: Getting Started" in response
        assert "Lesson 2: Basic Concepts" in response
        
        # Should still call with tools for potential outline tool use
        mock_ai_gen.return_value.generate_response.assert_called_once()
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_general_knowledge_query_no_tools(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test general knowledge query that shouldn't use tools"""
        # Arrange
        rag = RAGSystem(test_config)
        
        mock_ai_gen.return_value.generate_response.return_value = "The capital of France is Paris."
        mock_session.return_value.get_conversation_history.return_value = None
        mock_session.return_value.add_exchange = Mock()
        
        rag.tool_manager.get_last_sources = Mock(return_value=[])
        rag.tool_manager.reset_sources = Mock()
        
        # Act
        response, sources = rag.query("What is the capital of France?")
        
        # Assert
        assert response == "The capital of France is Paris."
        assert len(sources) == 0
        
        # Even general queries get tools provided (AI decides whether to use them)
        mock_ai_gen.return_value.generate_response.assert_called_once()


class TestRAGSystemSessionManagement:
    """Test RAG system session and conversation history management"""
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_with_session_id(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test query processing with session management"""
        # Arrange
        rag = RAGSystem(test_config)
        
        conversation_history = "Previous: What is MCP?\nResponse: MCP is a protocol..."
        mock_session.return_value.get_conversation_history.return_value = conversation_history
        mock_session.return_value.add_exchange = Mock()
        
        mock_ai_gen.return_value.generate_response.return_value = "Building on our previous discussion..."
        
        rag.tool_manager.get_last_sources = Mock(return_value=[])
        rag.tool_manager.reset_sources = Mock()
        
        session_id = "test_session_123"
        query = "Tell me more about it"
        
        # Act
        response, sources = rag.query(query, session_id=session_id)
        
        # Assert
        assert response == "Building on our previous discussion..."
        
        # Verify session operations
        mock_session.return_value.get_conversation_history.assert_called_once_with(session_id)
        mock_session.return_value.add_exchange.assert_called_once_with(session_id, query, response)
        
        # Verify history was passed to AI
        call_args = mock_ai_gen.return_value.generate_response.call_args
        assert call_args[1]['conversation_history'] == conversation_history
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_without_session_id(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test query processing without session management"""
        # Arrange
        rag = RAGSystem(test_config)
        
        mock_ai_gen.return_value.generate_response.return_value = "Direct response without context"
        
        rag.tool_manager.get_last_sources = Mock(return_value=[])
        rag.tool_manager.reset_sources = Mock()
        
        # Act
        response, sources = rag.query("What is MCP?")
        
        # Assert
        assert response == "Direct response without context"
        
        # Verify no session operations
        mock_session.return_value.get_conversation_history.assert_not_called()
        mock_session.return_value.add_exchange.assert_not_called()
        
        # Verify no history was passed to AI
        call_args = mock_ai_gen.return_value.generate_response.call_args
        assert call_args[1]['conversation_history'] is None


class TestRAGSystemDocumentProcessing:
    """Test RAG system document processing functionality"""
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_add_course_document_success(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test successful course document processing"""
        # Arrange
        rag = RAGSystem(test_config)
        
        # Mock document processor
        sample_course = Course(
            title="Test Course",
            instructor="Dr. Test",
            lessons=[Lesson(lesson_number=1, title="Test Lesson")]
        )
        sample_chunks = [
            CourseChunk(content="Test content", course_title="Test Course", chunk_index=0)
        ]
        
        mock_doc_proc.return_value.process_course_document.return_value = (sample_course, sample_chunks)
        
        # Mock vector store
        mock_vector_store.return_value.add_course_metadata = Mock()
        mock_vector_store.return_value.add_course_content = Mock()
        
        file_path = "/test/course.pdf"
        
        # Act
        course, chunk_count = rag.add_course_document(file_path)
        
        # Assert
        assert course.title == "Test Course"
        assert chunk_count == 1
        
        # Verify document processing
        mock_doc_proc.return_value.process_course_document.assert_called_once_with(file_path)
        
        # Verify vector store operations
        mock_vector_store.return_value.add_course_metadata.assert_called_once_with(sample_course)
        mock_vector_store.return_value.add_course_content.assert_called_once_with(sample_chunks)
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_add_course_document_error(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test error handling in document processing"""
        # Arrange
        rag = RAGSystem(test_config)
        
        mock_doc_proc.return_value.process_course_document.side_effect = Exception("Processing error")
        
        # Act
        course, chunk_count = rag.add_course_document("/invalid/path.pdf")
        
        # Assert
        assert course is None
        assert chunk_count == 0
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    @patch('os.path.exists')
    @patch('os.path.isfile')
    @patch('os.listdir')
    def test_add_course_folder_success(self, mock_listdir, mock_isfile, mock_exists, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test successful course folder processing"""
        # Arrange
        rag = RAGSystem(test_config)
        
        # Mock file system
        mock_exists.return_value = True
        mock_isfile.return_value = True  # Mock all files as valid files
        mock_listdir.return_value = ["course1.pdf", "course2.docx", "readme.txt", "course3.pdf"]
        
        # Mock existing courses
        mock_vector_store.return_value.get_existing_course_titles.return_value = ["Existing Course"]
        
        # Mock document processing
        courses_and_chunks = [
            (Course(title="Course 1"), [CourseChunk(content="Content 1", course_title="Course 1", chunk_index=0)]),
            (Course(title="Course 2"), [CourseChunk(content="Content 2", course_title="Course 2", chunk_index=0)]),
            (Course(title="Existing Course"), []),  # This should be skipped
            (Course(title="Course 3"), [CourseChunk(content="Content 3", course_title="Course 3", chunk_index=0)])
        ]
        
        mock_doc_proc.return_value.process_course_document.side_effect = courses_and_chunks
        
        # Mock vector store operations
        mock_vector_store.return_value.add_course_metadata = Mock()
        mock_vector_store.return_value.add_course_content = Mock()
        
        folder_path = "/test/courses"
        
        # Act
        total_courses, total_chunks = rag.add_course_folder(folder_path)
        
        # Assert
        assert total_courses == 3  # Should process 3 new courses (skip existing)
        assert total_chunks == 3   # 1 chunk per course
        
        # Verify vector store was called for new courses only
        assert mock_vector_store.return_value.add_course_metadata.call_count == 3
        assert mock_vector_store.return_value.add_course_content.call_count == 3
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    @patch('os.path.exists')
    def test_add_course_folder_nonexistent(self, mock_exists, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test handling of nonexistent folder"""
        # Arrange
        rag = RAGSystem(test_config)
        mock_exists.return_value = False
        
        # Act
        total_courses, total_chunks = rag.add_course_folder("/nonexistent/path")
        
        # Assert
        assert total_courses == 0
        assert total_chunks == 0


class TestRAGSystemAnalytics:
    """Test RAG system analytics and metadata functions"""
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_get_course_analytics(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test course analytics retrieval"""
        # Arrange
        rag = RAGSystem(test_config)
        
        mock_vector_store.return_value.get_course_count.return_value = 3
        mock_vector_store.return_value.get_existing_course_titles.return_value = [
            "Introduction to MCP", 
            "Advanced Python Programming",
            "Machine Learning Basics"
        ]
        
        # Act
        analytics = rag.get_course_analytics()
        
        # Assert
        assert analytics["total_courses"] == 3
        assert len(analytics["course_titles"]) == 3
        assert "Introduction to MCP" in analytics["course_titles"]


class TestRAGSystemErrorHandling:
    """Test RAG system error handling in various scenarios"""
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_ai_generator_error_handling(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test handling of AI generator errors"""
        # Arrange
        rag = RAGSystem(test_config)
        
        mock_ai_gen.return_value.generate_response.side_effect = Exception("AI API Error")
        mock_session.return_value.get_conversation_history.return_value = None
        
        # Act & Assert
        with pytest.raises(Exception, match="AI API Error"):
            rag.query("What is MCP?")
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_tool_manager_error_graceful_handling(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store, test_config):
        """Test graceful handling of tool manager errors"""
        # Arrange
        rag = RAGSystem(test_config)
        
        # Mock AI response
        mock_ai_gen.return_value.generate_response.return_value = "Fallback response due to tool error"
        
        # Mock tool manager errors
        rag.tool_manager.get_last_sources = Mock(side_effect=Exception("Tool error"))
        rag.tool_manager.reset_sources = Mock()
        
        mock_session.return_value.get_conversation_history.return_value = None
        mock_session.return_value.add_exchange = Mock()
        
        # Act
        response, sources = rag.query("What is MCP?")
        
        # Assert
        assert response == "Fallback response due to tool error"
        # Should gracefully handle tool error and return empty sources
        assert sources == []