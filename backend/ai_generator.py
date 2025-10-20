import anthropic
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class ConversationState:
    """Manages state across sequential tool calling rounds"""
    initial_query: str
    conversation_history: Optional[str]
    tools: Optional[List]
    tool_manager: Optional[object]
    messages: List[Dict[str, Any]]
    tool_results: List[str]
    round_number: int
    final_response: Optional[str]
    
    @classmethod
    def create(cls, query: str, conversation_history: Optional[str] = None,
               tools: Optional[List] = None, tool_manager=None):
        return cls(
            initial_query=query,
            conversation_history=conversation_history,
            tools=tools,
            tool_manager=tool_manager,
            messages=[{"role": "user", "content": query}],
            tool_results=[],
            round_number=0,
            final_response=None
        )
    
    def add_round_result(self, response, tool_results: List[str], tool_use_ids: List[str]):
        """Add results from a completed round"""
        self.round_number += 1
        self.tool_results.extend(tool_results)
        # Add assistant response to message history
        # Convert response.content to proper format for messages API
        content_blocks = []
        for content_block in response.content:
            if hasattr(content_block, 'type'):
                if content_block.type == "text":
                    content_blocks.append({
                        "type": "text",
                        "text": content_block.text
                    })
                elif content_block.type == "tool_use":
                    content_blocks.append({
                        "type": "tool_use",
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input
                    })
        self.messages.append({"role": "assistant", "content": content_blocks})
        # Add tool results if any
        if tool_results and tool_use_ids:
            tool_result_content = []
            for result, tool_use_id in zip(tool_results, tool_use_ids):
                tool_result_content.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result
                })
            self.messages.append({"role": "user", "content": tool_result_content})
    
    def set_final_response(self, response: str):
        """Set the final response text"""
        self.final_response = response
    
    def get_final_response(self) -> str:
        """Get the final response, defaulting to last response if not set"""
        return self.final_response or "I apologize, but I encountered an error processing your request."
    
    def get_system_content(self, round_num: int, max_rounds: int) -> str:
        """Build system content for the current round"""
        base_prompt = AIGenerator.SEQUENTIAL_SYSTEM_PROMPT
        
        # Add conversation history if available
        if self.conversation_history:
            base_prompt += f"\n\nPrevious conversation:\n{self.conversation_history}"
        
        # Add round-specific context
        round_context = f"\n\nCURRENT ROUND: {round_num}/{max_rounds}"
        
        if round_num == 1:
            round_context += "\nThis is your first round. Use tools strategically for information gathering."
        elif round_num == max_rounds:
            round_context += "\nThis is your final round. Synthesize information and provide a complete answer."
            if self.tool_results:
                round_context += f"\n\nPrevious tool results summary:\n{self._summarize_tool_results()}"
        else:
            round_context += "\nContinue gathering information or refine your search based on previous results."
            if self.tool_results:
                round_context += f"\n\nPrevious tool results:\n{self._summarize_tool_results()}"
        
        return base_prompt + round_context
    
    def _summarize_tool_results(self) -> str:
        """Create a summary of tool results for context"""
        if not self.tool_results:
            return "No tool results yet."
        
        summary = []
        for i, result in enumerate(self.tool_results, 1):
            # Truncate long results for context efficiency
            truncated = result[:300] + "..." if len(result) > 300 else result
            summary.append(f"Result {i}: {truncated}")
        
        return "\n".join(summary)

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage Guidelines:
- **Course Outline Queries**: Use `get_course_outline` for questions about course structure, lesson lists, or course overviews
- **Content Search Queries**: Use `search_course_content` for questions about specific course content or detailed educational materials
- **One tool call per query maximum**
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course structure questions** (outline, lessons, structure): Use course outline tool first, then answer
- **Course content questions** (specific topics, details): Use content search tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the tool results" or "according to the search"

When providing course outlines, include:
- Course title and instructor (if available)
- Course link (if available)
- Complete lesson list formatted as "Lesson X: Title"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    # Sequential system prompt for multi-round tool calling
    SEQUENTIAL_SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for multi-step reasoning.

**Sequential Reasoning Protocol:**
- You have up to 2 rounds of tool usage to answer complex questions
- Each tool call is a separate API interaction where you can reason about previous results
- Use round 1 for information gathering, round 2 for follow-up searches or clarification
- After tool usage is complete, provide your comprehensive final answer

**Tool Usage Strategy:**
- **Complex queries requiring multiple searches**: Use multiple rounds strategically
- **Simple queries**: Use tools once then answer directly  
- **Cross-referencing**: Get course outline first, then search specific content
- **Topic correlation**: Search course A for topic, then find courses with similar content

**Reasoning Examples:**
- "Find courses similar to lesson 4 of course X": 
  Round 1: Get course X outline to identify lesson 4 topic
  Round 2: Search for courses containing that topic
- "Compare lesson 3 content between courses A and B":
  Round 1: Search lesson 3 in course A  
  Round 2: Search lesson 3 in course B

**Termination Rules:**
- Provide final answer when you have sufficient information
- Don't use tools if previous results fully answer the question
- Quality over quantity - use tools purposefully

**Response Guidelines:**
- **No meta-commentary**: Don't mention "based on tool results" or reasoning process
- **Direct answers**: Get straight to the point
- **Educational**: Maintain instructional value
- **Clear**: Use accessible language

When providing course outlines, include:
- Course title and instructor (if available)
- Course link (if available)
- Complete lesson list formatted as "Lesson X: Title"
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response_sequential(self, query: str,
                                   conversation_history: Optional[str] = None,
                                   tools: Optional[List] = None,
                                   tool_manager=None,
                                   max_rounds: int = 2) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default 2)
            
        Returns:
            Generated response as string
        """
        # Create conversation state for this query
        conversation = ConversationState.create(
            query=query,
            conversation_history=conversation_history,
            tools=tools,
            tool_manager=tool_manager
        )
        
        try:
            # Execute sequential rounds
            for round_num in range(1, max_rounds + 1):
                response = self._execute_round(conversation, round_num, max_rounds)
                
                if self._should_terminate(response, round_num, max_rounds):
                    break
            
            # If we ended with tool use at max rounds, make one final call without tools
            if (response and response.stop_reason == "tool_use" and 
                conversation.round_number >= max_rounds and not conversation.final_response):
                final_response = self._execute_final_response(conversation)
                if final_response and final_response.content:
                    conversation.set_final_response(final_response.content[0].text)
            
            return conversation.get_final_response()
            
        except Exception as e:
            print(f"Error in sequential response generation: {e}")
            # Fallback to direct response if available
            if conversation.final_response:
                return conversation.final_response
            return "I apologize, but I encountered an error processing your request."
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Uses the original single-round approach for backwards compatibility.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        try:
            # Build system content
            system_content = self.SYSTEM_PROMPT
            if conversation_history:
                system_content += f"\\n\\nPrevious conversation:\\n{conversation_history}"
            
            # Prepare base API call parameters
            messages = [{"role": "user", "content": query}]
            base_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }
            
            # Add tools if provided
            if tools:
                base_params["tools"] = tools
                base_params["tool_choice"] = {"type": "auto"}
            
            # Make initial API call
            initial_response = self.client.messages.create(**base_params)
            
            # Handle tool execution if needed
            if initial_response.stop_reason == "tool_use" and tool_manager:
                return self._handle_tool_execution(initial_response, base_params, tool_manager)
            else:
                # No tool use, return direct response
                if initial_response.content:
                    return initial_response.content[0].text
                else:
                    return "I apologize, but I received an empty response."
                    
        except Exception as e:
            print(f"Error in AI response generation: {e}")
            return "I apologize, but I encountered an error processing your request."
    
    def _execute_round(self, conversation: ConversationState, round_num: int, max_rounds: int):
        """
        Execute a single round of the conversation with potential tool usage.
        
        Args:
            conversation: Current conversation state
            round_num: Current round number (1-based)
            max_rounds: Maximum number of rounds
            
        Returns:
            Response object from Claude API
        """
        try:
            # Build system content for this round
            system_content = conversation.get_system_content(round_num, max_rounds)
            
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": conversation.messages.copy(),
                "system": system_content
            }
            
            # Add tools if available (keep tools available across rounds)
            if conversation.tools:
                api_params["tools"] = conversation.tools
                api_params["tool_choice"] = {"type": "auto"}
            
            # Get response from Claude
            response = self.client.messages.create(**api_params)
            
            # Handle tool execution if needed
            if response.stop_reason == "tool_use" and conversation.tool_manager:
                tool_results, tool_use_ids = self._execute_tools_for_round(response, conversation.tool_manager)
                conversation.add_round_result(response, tool_results, tool_use_ids)
                return response
            else:
                # No tool use - this is the final response
                if response.content:
                    final_text = response.content[0].text
                    conversation.set_final_response(final_text)
                else:
                    conversation.set_final_response("I apologize, but I received an empty response.")
                return response
                
        except Exception as e:
            print(f"Error in round {round_num}: {e}")
            # Set error fallback response
            if round_num == 1:
                conversation.set_final_response("I apologize, but I encountered an error processing your request.")
            # For later rounds, we might have partial results to use
            return None
    
    def _should_terminate(self, response, round_num: int, max_rounds: int) -> bool:
        """
        Determine if we should terminate the conversation rounds.
        
        Args:
            response: The API response from the current round
            round_num: Current round number
            max_rounds: Maximum allowed rounds
            
        Returns:
            True if conversation should terminate, False to continue
        """
        # Terminate if we've reached max rounds
        if round_num >= max_rounds:
            return True
        
        # Terminate if response is None (error occurred)
        if response is None:
            return True
        
        # Terminate if Claude didn't use tools (provided direct answer)
        if response.stop_reason != "tool_use":
            return True
        
        # Continue for tool use (let next round decide)
        return False
    
    def _execute_tools_for_round(self, response, tool_manager):
        """
        Execute all tool calls from a response and collect results.
        
        Args:
            response: Claude's response containing tool use blocks
            tool_manager: Tool manager to execute tools
            
        Returns:
            Tuple of (tool_results: List[str], tool_use_ids: List[str])
        """
        tool_results = []
        tool_use_ids = []
        
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    tool_results.append(tool_result)
                    tool_use_ids.append(content_block.id)
                except Exception as e:
                    error_msg = f"Error executing tool {content_block.name}: {str(e)}"
                    print(error_msg)
                    tool_results.append(error_msg)
                    tool_use_ids.append(content_block.id)  # Still need to match the ID
        
        return tool_results, tool_use_ids
    
    def _execute_final_response(self, conversation: ConversationState):
        """
        Execute a final API call without tools to get Claude's synthesis.
        
        Args:
            conversation: Current conversation state
            
        Returns:
            Response object from Claude API or None if error
        """
        try:
            # Build system content indicating this is the final response
            # Build system content indicating this is the final response
            system_content = conversation.get_system_content(
                conversation.round_number + 1, conversation.round_number + 1
            )
            system_content += "\n\nIMPORTANT: This is your final response. Provide a complete answer based on the tool results you have gathered. Do not request more tools."
            
            # Prepare API call without tools
            api_params = {
                **self.base_params,
                "messages": conversation.messages.copy(),
                "system": system_content
                # Note: No tools provided for final response
            }
            
            # Get final response from Claude
            response = self.client.messages.create(**api_params)
            return response
            
        except Exception as e:
            print(f"Error in final response: {e}")
            return None
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        if final_response.content:
            return final_response.content[0].text
        else:
            return "I apologize, but I received an empty response."