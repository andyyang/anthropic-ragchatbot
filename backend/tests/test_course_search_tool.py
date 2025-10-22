"""
Unit tests for CourseSearchTool.execute method.
Tests the core functionality of semantic course content search.
"""

import os
import sys

# Add backend to path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Test suite for CourseSearchTool.execute method"""

    def test_execute_basic_query_success(self, mock_vector_store, mock_search_results):
        """Test basic query execution with successful results"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results("success")

        # Act
        result = tool.execute("What is MCP?")

        # Assert
        assert isinstance(result, str)
        assert "Introduction to MCP" in result
        assert "MCP (Model Context Protocol)" in result
        assert len(tool.last_sources) == 2
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name=None, lesson_number=None
        )

    def test_execute_with_course_filter(self, mock_vector_store, mock_search_results):
        """Test query execution with course name filtering"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results("course_filtered")

        # Act
        result = tool.execute("decorators", course_name="Python")

        # Assert
        assert "Advanced Python Programming" in result
        assert "decorators" in result
        assert len(tool.last_sources) == 1
        mock_vector_store.search.assert_called_once_with(
            query="decorators", course_name="Python", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store, mock_search_results):
        """Test query execution with lesson number filtering"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results("success")

        # Act
        result = tool.execute("basic concepts", lesson_number=2)

        # Assert
        assert "Lesson 2" in result
        assert len(tool.last_sources) == 2
        mock_vector_store.search.assert_called_once_with(
            query="basic concepts", course_name=None, lesson_number=2
        )

    def test_execute_with_both_filters(self, mock_vector_store, mock_search_results):
        """Test query execution with both course and lesson filtering"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        filtered_result = SearchResults(
            documents=["Specific lesson content about MCP"],
            metadata=[{"course_title": "Introduction to MCP", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = filtered_result

        # Act
        result = tool.execute("MCP basics", course_name="MCP", lesson_number=1)

        # Assert
        assert "[Introduction to MCP - Lesson 1]" in result
        assert "Specific lesson content" in result
        assert len(tool.last_sources) == 1
        mock_vector_store.search.assert_called_once_with(
            query="MCP basics", course_name="MCP", lesson_number=1
        )

    def test_execute_empty_results(self, mock_vector_store, mock_search_results):
        """Test handling of empty search results"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results("empty")

        # Act
        result = tool.execute("nonexistent topic")

        # Assert
        assert "No relevant content found" in result
        assert len(tool.last_sources) == 0

    def test_execute_empty_results_with_filters(
        self, mock_vector_store, mock_search_results
    ):
        """Test empty results with filter information included"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results("empty")

        # Act
        result = tool.execute("topic", course_name="NonExistent", lesson_number=5)

        # Assert
        assert "No relevant content found in course 'NonExistent' in lesson 5" in result
        assert len(tool.last_sources) == 0

    def test_execute_vector_store_error(self, mock_vector_store, mock_search_results):
        """Test handling of vector store errors"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_search_results("error")

        # Act
        result = tool.execute("any query")

        # Assert
        assert "Vector store connection failed" in result
        assert len(tool.last_sources) == 0

    def test_execute_source_tracking(self, mock_vector_store):
        """Test that sources are properly tracked and formatted"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        search_result = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": None},
            ],
            distances=[0.1, 0.2],
        )
        mock_vector_store.search.return_value = search_result
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        # Act
        tool.execute("test query")

        # Assert
        assert len(tool.last_sources) == 2

        # Check first source (with lesson)
        assert tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com/lesson1"

        # Check second source (without lesson)
        assert tool.last_sources[1]["text"] == "Course B"
        assert tool.last_sources[1]["url"] is None

    def test_execute_result_formatting(self, mock_vector_store):
        """Test that results are properly formatted with headers"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        search_result = SearchResults(
            documents=["First content piece", "Second content piece"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Another Course", "lesson_number": None},
            ],
            distances=[0.1, 0.2],
        )
        mock_vector_store.search.return_value = search_result

        # Act
        result = tool.execute("test")

        # Assert
        assert "[Test Course - Lesson 1]" in result
        assert "[Another Course]" in result
        assert "First content piece" in result
        assert "Second content piece" in result

        # Check that results are separated properly
        parts = result.split("\n\n")
        assert len(parts) == 2

    def test_execute_metadata_edge_cases(self, mock_vector_store):
        """Test handling of edge cases in metadata"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        search_result = SearchResults(
            documents=["Content with missing metadata"],
            metadata=[{"course_title": None, "lesson_number": None}],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = search_result

        # Act
        result = tool.execute("test")

        # Assert
        assert "[unknown]" in result  # Should handle None course_title
        assert "Content with missing metadata" in result
        assert len(tool.last_sources) == 1

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly formatted"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        definition = tool.get_tool_definition()

        # Assert
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]

        properties = definition["input_schema"]["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties


class TestCourseSearchToolIntegration:
    """Integration tests for CourseSearchTool with ToolManager"""

    def test_tool_manager_registration(self, mock_vector_store):
        """Test that CourseSearchTool can be registered with ToolManager"""
        # Arrange
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)

        # Act
        tool_manager.register_tool(search_tool)

        # Assert
        definitions = tool_manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_tool_execution_via_manager(self, mock_vector_store, mock_search_results):
        """Test tool execution through ToolManager"""
        # Arrange
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        mock_vector_store.search.return_value = mock_search_results("success")

        # Act
        result = tool_manager.execute_tool(
            "search_course_content", query="test query", course_name="Test Course"
        )

        # Assert
        assert isinstance(result, str)
        assert "Introduction to MCP" in result
        mock_vector_store.search.assert_called_once()

    def test_source_retrieval_via_manager(self, mock_vector_store, mock_search_results):
        """Test that sources can be retrieved through ToolManager"""
        # Arrange
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        mock_vector_store.search.return_value = mock_search_results("success")

        # Act
        tool_manager.execute_tool("search_course_content", query="test")
        sources = tool_manager.get_last_sources()

        # Assert
        assert len(sources) > 0
        assert all("text" in source for source in sources)

    def test_source_reset_via_manager(self, mock_vector_store, mock_search_results):
        """Test that sources can be reset through ToolManager"""
        # Arrange
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        mock_vector_store.search.return_value = mock_search_results("success")

        # Act
        tool_manager.execute_tool("search_course_content", query="test")
        assert len(tool_manager.get_last_sources()) > 0

        tool_manager.reset_sources()
        sources = tool_manager.get_last_sources()

        # Assert
        assert len(sources) == 0
