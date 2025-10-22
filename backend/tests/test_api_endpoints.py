"""
API endpoint tests for the RAG system FastAPI application.
Tests all API endpoints for proper request/response handling, error scenarios,
and integration with the RAG system components.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from fastapi.testclient import TestClient


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint for processing user queries"""
    
    def test_query_without_session_id(self, test_client, mock_rag_system):
        """Test query endpoint creates new session when none provided"""
        # Arrange
        request_data = {"query": "What is MCP?"}
        
        # Act
        response = test_client.post("/api/query", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "This is a test response from the RAG system."
        assert data["session_id"] == "test_session_123"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Test Source - Lesson 1"
        assert data["sources"][0]["url"] == "https://example.com/test/lesson1"
        
        # Verify RAG system was called correctly
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is MCP?", "test_session_123")
    
    def test_query_with_existing_session_id(self, test_client, mock_rag_system):
        """Test query endpoint uses provided session ID"""
        # Arrange
        request_data = {
            "query": "Tell me more about it",
            "session_id": "existing_session_456"
        }
        
        # Act
        response = test_client.post("/api/query", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "existing_session_456"
        
        # Verify existing session was used (no new session created)
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("Tell me more about it", "existing_session_456")
    
    def test_query_with_string_sources_backward_compatibility(self, test_app):
        """Test query endpoint handles old string-format sources"""
        # Arrange
        mock_rag = Mock()
        mock_rag.query.return_value = (
            "Response with string sources",
            ["String source 1", "String source 2"]  # Old format
        )
        mock_rag.session_manager.create_session.return_value = "test_session"
        
        test_app.state.rag_system = mock_rag
        test_client = TestClient(test_app)
        request_data = {"query": "Test query"}
        
        # Act
        response = test_client.post("/api/query", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["sources"]) == 2
        assert data["sources"][0]["text"] == "String source 1"
        assert data["sources"][0]["url"] is None
        assert data["sources"][1]["text"] == "String source 2"
        assert data["sources"][1]["url"] is None
    
    def test_query_missing_query_field(self, test_client):
        """Test query endpoint returns error for missing query field"""
        # Arrange
        request_data = {"session_id": "test_session"}  # Missing query field
        
        # Act
        response = test_client.post("/api/query", json=request_data)
        
        # Assert
        assert response.status_code == 422  # Validation error
        error_data = response.json()
        assert "detail" in error_data
        assert any("query" in str(error).lower() for error in error_data["detail"])
    
    def test_query_empty_query_string(self, test_client, mock_rag_system):
        """Test query endpoint handles empty query string"""
        # Arrange
        request_data = {"query": ""}
        
        # Act
        response = test_client.post("/api/query", json=request_data)
        
        # Assert
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with("", "test_session_123")
    
    def test_query_rag_system_error_handling(self, test_app):
        """Test query endpoint handles RAG system errors gracefully"""
        # Arrange
        mock_rag = Mock()
        mock_rag.query.side_effect = Exception("RAG system error")
        mock_rag.session_manager.create_session.return_value = "test_session"
        
        test_app.state.rag_system = mock_rag
        test_client = TestClient(test_app)
        request_data = {"query": "What is MCP?"}
        
        # Act
        response = test_client.post("/api/query", json=request_data)
        
        # Assert
        assert response.status_code == 500
        error_data = response.json()
        assert error_data["detail"] == "RAG system error"
    
    def test_query_invalid_json_format(self, test_client):
        """Test query endpoint handles invalid JSON"""
        # Act
        response = test_client.post("/api/query", data="invalid json")
        
        # Assert
        assert response.status_code == 422


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint for course analytics"""
    
    def test_get_course_stats_success(self, test_client, mock_rag_system):
        """Test successful course statistics retrieval"""
        # Act
        response = test_client.get("/api/courses")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Introduction to MCP" in data["course_titles"]
        assert "Advanced Python Programming" in data["course_titles"]
        
        # Verify RAG system was called
        mock_rag_system.get_course_analytics.assert_called_once()
    
    def test_get_course_stats_empty_courses(self, test_app):
        """Test course statistics when no courses exist"""
        # Arrange
        mock_rag = Mock()
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        test_app.state.rag_system = mock_rag
        test_client = TestClient(test_app)
        
        # Act
        response = test_client.get("/api/courses")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_get_course_stats_rag_system_error(self, test_app):
        """Test course statistics endpoint handles errors"""
        # Arrange
        mock_rag = Mock()
        mock_rag.get_course_analytics.side_effect = Exception("Database connection failed")
        
        test_app.state.rag_system = mock_rag
        test_client = TestClient(test_app)
        
        # Act
        response = test_client.get("/api/courses")
        
        # Assert
        assert response.status_code == 500
        error_data = response.json()
        assert error_data["detail"] == "Database connection failed"


@pytest.mark.api
class TestClearSessionEndpoint:
    """Test the /api/clear-session endpoint for session management"""
    
    def test_clear_session_success(self, test_client, mock_rag_system):
        """Test successful session clearing"""
        # Arrange
        request_data = {"session_id": "session_to_clear_123"}
        
        # Act
        response = test_client.post("/api/clear-session", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "Session cleared successfully"
        
        # Verify session manager was called
        mock_rag_system.session_manager.clear_session.assert_called_once_with("session_to_clear_123")
    
    def test_clear_session_missing_session_id(self, test_client):
        """Test clear session endpoint with missing session_id field"""
        # Arrange
        request_data = {}  # Missing session_id
        
        # Act
        response = test_client.post("/api/clear-session", json=request_data)
        
        # Assert
        assert response.status_code == 422  # Validation error
        error_data = response.json()
        assert "detail" in error_data
        assert any("session_id" in str(error).lower() for error in error_data["detail"])
    
    def test_clear_session_empty_session_id(self, test_client, mock_rag_system):
        """Test clear session endpoint with empty session_id"""
        # Arrange
        request_data = {"session_id": ""}
        
        # Act
        response = test_client.post("/api/clear-session", json=request_data)
        
        # Assert
        assert response.status_code == 200
        mock_rag_system.session_manager.clear_session.assert_called_once_with("")
    
    def test_clear_session_rag_system_error(self, test_app):
        """Test clear session endpoint handles RAG system errors"""
        # Arrange
        mock_rag = Mock()
        mock_rag.session_manager.clear_session.side_effect = Exception("Session not found")
        
        test_app.state.rag_system = mock_rag
        test_client = TestClient(test_app)
        request_data = {"session_id": "nonexistent_session"}
        
        # Act
        response = test_client.post("/api/clear-session", json=request_data)
        
        # Assert
        assert response.status_code == 500
        error_data = response.json()
        assert error_data["detail"] == "Session not found"


@pytest.mark.api
class TestRootEndpoint:
    """Test the root endpoint / for basic API health check"""
    
    def test_root_endpoint_success(self, test_client):
        """Test root endpoint returns expected message"""
        # Act
        response = test_client.get("/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "RAG System API - Test Environment"
    
    def test_root_endpoint_method_not_allowed(self, test_client):
        """Test root endpoint only accepts GET requests"""
        # Act
        response = test_client.post("/")
        
        # Assert
        assert response.status_code == 405  # Method Not Allowed


@pytest.mark.api
class TestAPIMiddleware:
    """Test API middleware configuration and CORS handling"""
    
    def test_cors_headers_present(self, test_client):
        """Test CORS middleware adds appropriate headers on cross-origin request"""
        # Act - Make a request with Origin header to trigger CORS
        response = test_client.get("/", headers={"Origin": "http://localhost:3000"})
        
        # Assert
        assert response.status_code == 200
        # CORS headers should be present when Origin header is provided
        headers_lower = [h.lower() for h in response.headers.keys()]
        assert "access-control-allow-origin" in headers_lower
    
    def test_options_request_handling(self, test_client):
        """Test CORS preflight OPTIONS request handling"""
        # Act - Make an OPTIONS request with proper CORS headers
        response = test_client.options(
            "/api/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type"
            }
        )
        
        # Assert
        assert response.status_code == 200
        headers = {k.lower(): v for k, v in response.headers.items()}
        assert "access-control-allow-origin" in headers


@pytest.mark.api
class TestAPIErrorScenarios:
    """Test various API error scenarios and edge cases"""
    
    def test_nonexistent_endpoint_404(self, test_client):
        """Test accessing nonexistent endpoint returns 404"""
        # Act
        response = test_client.get("/api/nonexistent")
        
        # Assert
        assert response.status_code == 404
        error_data = response.json()
        assert error_data["detail"] == "Not Found"
    
    def test_invalid_http_method_405(self, test_client):
        """Test using invalid HTTP method returns 405"""
        # Act
        response = test_client.patch("/api/courses")
        
        # Assert
        assert response.status_code == 405  # Method Not Allowed
    
    def test_large_request_payload(self, test_client, mock_rag_system):
        """Test handling of large request payload"""
        # Arrange
        large_query = "What is MCP? " * 1000  # Very large query
        request_data = {"query": large_query}
        
        # Act
        response = test_client.post("/api/query", json=request_data)
        
        # Assert
        assert response.status_code == 200
        # Should still process the request
        mock_rag_system.query.assert_called_once()
        call_args = mock_rag_system.query.call_args[0]
        assert call_args[0] == large_query


@pytest.mark.api
class TestAPIResponseFormats:
    """Test API response formats and data validation"""
    
    def test_query_response_schema_validation(self, test_client, mock_rag_system):
        """Test query response matches expected schema"""
        # Arrange
        request_data = {"query": "Test query"}
        
        # Act
        response = test_client.post("/api/query", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields exist and have correct types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Verify source structure
        if data["sources"]:
            source = data["sources"][0]
            assert "text" in source
            assert "url" in source
            assert isinstance(source["text"], str)
    
    def test_courses_response_schema_validation(self, test_client, mock_rag_system):
        """Test courses response matches expected schema"""
        # Act
        response = test_client.get("/api/courses")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields and types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert all(isinstance(title, str) for title in data["course_titles"])
        assert data["total_courses"] == len(data["course_titles"])


@pytest.mark.api 
@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints working together"""
    
    def test_session_continuity_across_queries(self, test_app):
        """Test session continuity across multiple queries"""
        # Arrange
        mock_rag = Mock()
        mock_rag.session_manager.create_session.return_value = "persistent_session_123"
        mock_rag.query.return_value = ("Response", [])
        
        test_app.state.rag_system = mock_rag
        test_client = TestClient(test_app)
        
        # Act - First query without session
        response1 = test_client.post("/api/query", json={"query": "First query"})
        session_id = response1.json()["session_id"]
        
        # Act - Second query with same session
        response2 = test_client.post("/api/query", json={
            "query": "Follow-up query",
            "session_id": session_id
        })
        
        # Assert
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id
        
        # Verify session was created once and reused
        mock_rag.session_manager.create_session.assert_called_once()
        assert mock_rag.query.call_count == 2
    
    def test_query_then_clear_session_workflow(self, test_app):
        """Test complete workflow: query -> clear session"""
        # Arrange
        mock_rag = Mock()
        mock_rag.session_manager.create_session.return_value = "workflow_session_456"
        mock_rag.query.return_value = ("Response", [])
        mock_rag.session_manager.clear_session = Mock()
        
        test_app.state.rag_system = mock_rag
        test_client = TestClient(test_app)
        
        # Act - Query to create session
        query_response = test_client.post("/api/query", json={"query": "Test query"})
        session_id = query_response.json()["session_id"]
        
        # Act - Clear the session
        clear_response = test_client.post("/api/clear-session", json={"session_id": session_id})
        
        # Assert
        assert query_response.status_code == 200
        assert clear_response.status_code == 200
        assert clear_response.json()["message"] == "Session cleared successfully"
        
        # Verify operations
        mock_rag.session_manager.create_session.assert_called_once()
        mock_rag.query.assert_called_once()
        mock_rag.session_manager.clear_session.assert_called_once_with(session_id)