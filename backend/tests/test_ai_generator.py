"""
Integration tests for AIGenerator with CourseSearchTool.
Tests the AI's ability to correctly call and integrate with search tools.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend to path  
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager
from .fixtures.sample_data import ANTHROPIC_RESPONSES, TEST_QUERIES


class TestAIGeneratorBasicFunctionality:
    """Test basic AIGenerator functionality without tools"""
    
    def test_init(self, test_config):
        """Test AIGenerator initialization"""
        # Act
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        # Assert
        assert generator.model == test_config.ANTHROPIC_MODEL
        assert generator.base_params["model"] == test_config.ANTHROPIC_MODEL
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic, test_config):
        """Test direct response generation without tool use"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Direct response from Claude")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        # Act
        result = generator.generate_response("What is the capital of France?")
        
        # Assert
        assert result == "Direct response from Claude"
        mock_client.messages.create.assert_called_once()
        
        # Verify API call parameters
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["messages"][0]["content"] == "What is the capital of France?"
        assert "tools" not in call_args[1]
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic, test_config):
        """Test response generation with conversation history"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with context")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        history = "Previous: What is MCP?\nResponse: MCP is a protocol..."
        
        # Act
        result = generator.generate_response("Tell me more", conversation_history=history)
        
        # Assert
        assert result == "Response with context"
        
        # Verify history was included in system prompt
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert history in system_content


class TestAIGeneratorToolIntegration:
    """Test AIGenerator integration with CourseSearchTool"""
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools_no_use(self, mock_anthropic, test_config, mock_tool_manager):
        """Test response when tools are available but not used"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Direct response without tools")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        tools = mock_tool_manager.get_tool_definitions()
        
        # Act
        result = generator.generate_response(
            "General knowledge question",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Assert
        assert result == "Direct response without tools"
        
        # Verify tools were provided to API
        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tool_choice"] == {"type": "auto"}
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic, test_config, mock_tool_manager):
        """Test response generation with tool use"""
        # Arrange
        mock_client = Mock()
        
        # First response with tool use
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_content = Mock()
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.input = {"query": "MCP basics", "course_name": "Introduction to MCP"}
        tool_content.id = "tool_123"
        tool_response.content = [tool_content]
        
        # Final response after tool execution
        final_response = Mock()
        final_response.content = [Mock(text="Based on search results, MCP is...")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool execution
        mock_tool_manager.execute_tool.return_value = "Search results about MCP"
        
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        tools = mock_tool_manager.get_tool_definitions()
        
        # Act
        result = generator.generate_response(
            "What is MCP?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Assert
        assert result == "Based on search results, MCP is..."
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="MCP basics",
            course_name="Introduction to MCP"
        )
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_error(self, mock_anthropic, test_config, mock_tool_manager):
        """Test handling of tool execution errors"""
        # Arrange
        mock_client = Mock()
        
        # Tool use response
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_content = Mock()
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.input = {"query": "invalid query"}
        tool_content.id = "tool_123"
        tool_response.content = [tool_content]
        
        # Final response after tool execution
        final_response = Mock()
        final_response.content = [Mock(text="I apologize, there was an error...")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool execution error
        mock_tool_manager.execute_tool.return_value = "Error: Tool execution failed"
        
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        tools = mock_tool_manager.get_tool_definitions()
        
        # Act
        result = generator.generate_response(
            "Invalid query",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Assert
        assert result == "I apologize, there was an error..."
        mock_tool_manager.execute_tool.assert_called_once()
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_multiple_tool_calls_in_single_response(self, mock_anthropic, test_config, mock_tool_manager):
        """Test handling multiple tool calls in a single response"""
        # Arrange  
        mock_client = Mock()
        
        # Response with multiple tool uses
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        
        tool_content_1 = Mock()
        tool_content_1.type = "tool_use"
        tool_content_1.name = "search_course_content"
        tool_content_1.input = {"query": "MCP basics"}
        tool_content_1.id = "tool_123"
        
        tool_content_2 = Mock()
        tool_content_2.type = "tool_use"
        tool_content_2.name = "get_course_outline"
        tool_content_2.input = {"course_title": "Introduction to MCP"}
        tool_content_2.id = "tool_456"
        
        tool_response.content = [tool_content_1, tool_content_2]
        
        # Final response
        final_response = Mock()
        final_response.content = [Mock(text="Combined results from multiple tools")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool executions
        mock_tool_manager.execute_tool.side_effect = [
            "Search results",
            "Course outline"
        ]
        
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        tools = mock_tool_manager.get_tool_definitions()
        
        # Act
        result = generator.generate_response(
            "Tell me about MCP course",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Assert
        assert result == "Combined results from multiple tools"
        assert mock_tool_manager.execute_tool.call_count == 2


class TestAIGeneratorCourseSearchIntegration:
    """Test specific integration with CourseSearchTool functionality"""
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_course_content_search_integration(self, mock_anthropic, test_config, mock_vector_store, mock_search_results):
        """Test integration with actual CourseSearchTool for content queries"""
        # Arrange
        mock_client = Mock()
        
        # Tool use response
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_content = Mock()
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.input = {"query": "MCP protocol", "course_name": "Introduction to MCP"}
        tool_content.id = "tool_123"
        tool_response.content = [tool_content]
        
        # Final response
        final_response = Mock()
        final_response.content = [Mock(text="MCP is a protocol for AI integration")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Setup real CourseSearchTool and ToolManager
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        mock_vector_store.search.return_value = mock_search_results("success")
        
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        # Act
        result = generator.generate_response(
            "What is MCP protocol?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Assert
        assert result == "MCP is a protocol for AI integration"
        
        # Verify vector store was called
        mock_vector_store.search.assert_called_once_with(
            query="MCP protocol",
            course_name="Introduction to MCP",
            lesson_number=None
        )
        
        # Verify sources were tracked
        sources = tool_manager.get_last_sources()
        assert len(sources) > 0
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_course_outline_integration(self, mock_anthropic, test_config, mock_vector_store):
        """Test integration with CourseOutlineTool"""
        # Arrange
        mock_client = Mock()
        
        # Tool use response
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_content = Mock()
        tool_content.type = "tool_use"
        tool_content.name = "get_course_outline"
        tool_content.input = {"course_title": "Introduction to MCP"}
        tool_content.id = "tool_123"
        tool_response.content = [tool_content]
        
        # Final response
        final_response = Mock()
        final_response.content = [Mock(text="The MCP course covers 3 lessons...")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Setup tool manager with outline tool
        from search_tools import CourseOutlineTool
        tool_manager = ToolManager()
        outline_tool = CourseOutlineTool(mock_vector_store)
        tool_manager.register_tool(outline_tool)
        
        # Mock course resolution and metadata
        mock_vector_store._resolve_course_name.return_value = "Introduction to MCP"
        mock_vector_store.course_catalog.get.return_value = {
            'metadatas': [{
                'title': 'Introduction to MCP',
                'instructor': 'Dr. Smith',
                'course_link': 'https://example.com/mcp',
                'lessons_json': '[{"lesson_number": 1, "lesson_title": "Getting Started"}]'
            }]
        }
        
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        # Act
        result = generator.generate_response(
            "What lessons are in MCP course?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Assert
        assert result == "The MCP course covers 3 lessons..."
        mock_vector_store._resolve_course_name.assert_called_once_with("Introduction to MCP")
        mock_vector_store.course_catalog.get.assert_called_once()


class TestAIGeneratorErrorHandling:
    """Test AIGenerator error handling scenarios"""
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_anthropic_api_error(self, mock_anthropic, test_config):
        """Test handling of Anthropic API errors"""
        # Arrange
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        
        # Act & Assert
        with pytest.raises(Exception, match="API Error"):
            generator.generate_response("Any query")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_manager_not_provided_with_tool_use(self, mock_anthropic, test_config):
        """Test behavior when tool use occurs but no tool manager provided"""
        # Arrange
        mock_client = Mock()
        
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_content = Mock()
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.input = {"query": "test"}
        tool_content.id = "tool_123"
        tool_response.content = [tool_content]
        
        mock_client.messages.create.return_value = tool_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        tools = [{"name": "test_tool", "description": "Test"}]
        
        # Act
        result = generator.generate_response("Query", tools=tools, tool_manager=None)
        
        # Assert - Should return the tool_use response directly since no tool manager
        assert hasattr(result, 'stop_reason')  # Returns the mock response object