from typing import Any, Dict, List, Optional, Tuple

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- You may search **up to 2 times** per query, in sequence. Search again only if the first results are insufficient or the question needs information from a different course or lesson (e.g. comparing courses, or finding a lesson's topic and then searching for related content)
- Each search must differ meaningfully from the previous one; never repeat a search
- Synthesize all search results into accurate, fact-based responses
- If searches yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search (once, or twice if needed), then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    # Sequential tool-use rounds per query; the call after the last round can't use tools
    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters (current models reject non-default temperature
        # and think before answering, so leave headroom beyond the answer itself)
        self.base_params = {"model": self.model, "max_tokens": 4096}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently. The API requires `tools` on every
        # call whose messages contain tool_use/tool_result blocks, so keep them throughout.
        api_params = {**self.base_params, "system": system_content}
        if tools:
            api_params["tools"] = tools
        messages = [{"role": "user", "content": query}]

        response = self._create(api_params, messages, tools, allow_tools=True)

        # Each round: run the requested tools, then let Claude see the results. After
        # MAX_TOOL_ROUNDS or a failed tool call, the next call must answer without tools.
        rounds = 0
        while (
            response.stop_reason == "tool_use"
            and tool_manager
            and rounds < self.MAX_TOOL_ROUNDS
        ):
            rounds += 1
            # Pass the content back verbatim; thinking blocks must stay intact
            messages.append({"role": "assistant", "content": response.content})
            tool_results, failed = self._execute_tools(response, tool_manager)
            messages.append({"role": "user", "content": tool_results})

            allow_tools = not failed and rounds < self.MAX_TOOL_ROUNDS
            response = self._create(api_params, messages, tools, allow_tools)

        return self._extract_text(response)

    def _create(
        self, api_params: Dict[str, Any], messages: List[Dict], tools, allow_tools: bool
    ):
        """Make one API call; tool_choice is auto while more tool rounds are allowed"""
        params = {**api_params, "messages": list(messages)}
        if tools:
            params["tool_choice"] = {"type": "auto" if allow_tools else "none"}
        return self.client.messages.create(**params)

    @staticmethod
    def _extract_text(response) -> str:
        """Join the text blocks of a response, skipping thinking and tool_use blocks"""
        return "".join(block.text for block in response.content if block.type == "text")

    @staticmethod
    def _execute_tools(response, tool_manager) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Execute every tool_use block in a response.

        Args:
            response: The response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            (tool_result blocks for a single user message, whether any tool raised)
        """
        tool_results = []
        failed = False
        for content_block in response.content:
            if content_block.type != "tool_use":
                continue
            result = {"type": "tool_result", "tool_use_id": content_block.id}
            try:
                result["content"] = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )
            except Exception as e:
                # Every tool_use still needs a tool_result; report the error to Claude
                result["content"] = f"Tool error: {e}"
                result["is_error"] = True
                failed = True
            tool_results.append(result)
        return tool_results, failed
