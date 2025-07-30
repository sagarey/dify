"""
OpenAI API Chat Large Language Model Implementation

This module implements the LLM interface for OpenAI-compatible chat completion APIs.
It specifically targets the /chat/completions endpoint and provides:
- Streaming and non-streaming completions
- Function calling support
- Multi-turn conversation handling
- Proper message formatting
"""

import json
import logging
from collections.abc import Generator
from typing import Optional, Union

import httpx

from core.model_runtime.entities.llm_entities import LLMResult, LLMResultChunk, LLMResultChunkDelta, LLMUsage
from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageContentUnionTypes,
    PromptMessageTool,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)
# Remove unused ModelType import
from core.model_runtime.errors.invoke import (
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from core.model_runtime.errors.validate import CredentialsValidateFailedError
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel

logger = logging.getLogger(__name__)


class OpenAIChatLargeLanguageModel(LargeLanguageModel):
    """
    OpenAI API Chat Large Language Model - specifically for /chat/completions endpoint
    
    This implementation focuses exclusively on chat-based completions,
    providing optimized support for conversation-style interactions.
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: Optional[dict] = None,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model for chat completions

        Args:
            model: Model name
            credentials: Model credentials
            prompt_messages: Prompt messages for conversation
            model_parameters: Model parameters
            tools: Tools for function calling
            stop: Stop words
            stream: Whether to stream response
            user: Unique user identifier

        Returns:
            LLMResult for non-streaming, Generator for streaming
        """
        # Validate that we're using chat mode
        if credentials.get('mode', 'chat') != 'chat':
            raise CredentialsValidateFailedError(
                'This provider only supports chat mode'
            )

        return self._chat_completion_request(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters or {},
            tools=tools,
            stop=stop,
            stream=stream,
            user=user,
        )

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages

        Args:
            model: Model name
            credentials: Model credentials
            prompt_messages: Prompt messages
            tools: Tools for function calling

        Returns:
            Estimated number of tokens
        """
        # Simple token estimation - in production, use proper tokenizer
        total_content = ""
        
        for message in prompt_messages:
            if isinstance(message, (SystemPromptMessage, UserPromptMessage, AssistantPromptMessage)):
                if isinstance(message.content, str):
                    total_content += message.content
                elif isinstance(message.content, list):
                    for content_part in message.content:
                        if hasattr(content_part, 'data') and isinstance(content_part.data, str):
                            total_content += content_part.data
            elif isinstance(message, ToolPromptMessage):
                total_content += str(message.content)
        
        if tools:
            for tool in tools:
                total_content += tool.name + str(tool.description) + str(tool.parameters)
        
        # Rough estimation: 1 token ≈ 4 characters
        return max(len(total_content) // 4, 1)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials by making a test request

        Args:
            model: Model name
            credentials: Model credentials

        Raises:
            CredentialsValidateFailedError: If credentials are invalid
        """
        try:
            # Make a minimal test request
            self._chat_completion_request(
                model=model,
                credentials=credentials,
                prompt_messages=[
                    SystemPromptMessage(content="You are a helpful assistant."),
                    UserPromptMessage(content="Hello")
                ],
                model_parameters={"max_tokens": 5},
                stream=False,
            )
        except Exception as ex:
            raise CredentialsValidateFailedError(f"Credential validation failed: {str(ex)}")

    def _chat_completion_request(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Make chat completion request to OpenAI-compatible API

        Args:
            model: Model name
            credentials: API credentials
            prompt_messages: Conversation messages
            model_parameters: Model parameters
            tools: Available tools
            stop: Stop sequences
            stream: Stream response
            user: User identifier

        Returns:
            LLMResult or streaming generator
        """
        api_key = credentials.get('api_key')
        endpoint_url = credentials.get('endpoint_url', 'https://api.openai.com/v1')
        
        if not api_key:
            raise CredentialsValidateFailedError('API Key is required')

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

        # Convert Dify messages to OpenAI format
        messages = self._convert_messages_to_openai_format(prompt_messages)

        # Prepare request payload
        payload = {
            'model': model,
            'messages': messages,
            'stream': stream,
        }

        # Add model parameters
        payload.update(model_parameters)

        # Add function calling tools if provided
        if tools:
            payload['tools'] = self._convert_tools_to_openai_format(tools)

        # Add stop sequences
        if stop:
            payload['stop'] = stop

        # Add user identifier
        if user:
            payload['user'] = user

        # Ensure we don't exceed context limits
        context_size = int(credentials.get('context_size', 4096))
        if 'max_tokens' not in payload:
            # Reserve some tokens for the response
            estimated_prompt_tokens = self.get_num_tokens(model, credentials, prompt_messages, tools)
            max_response_tokens = min(
                int(credentials.get('max_tokens', 2048)),
                max(context_size - estimated_prompt_tokens - 100, 100)
            )
            payload['max_tokens'] = max_response_tokens

        # Make API request
        api_url = self._get_chat_completions_url(endpoint_url)
        
        try:
            if stream:
                return self._handle_streaming_response(api_url, headers, payload, model, prompt_messages)
            else:
                return self._handle_non_streaming_response(api_url, headers, payload, model, prompt_messages)
                
        except httpx.ConnectError as ex:
            raise InvokeConnectionError(f"Connection failed: {str(ex)}")
        except httpx.TimeoutException as ex:
            raise InvokeConnectionError(f"Request timeout: {str(ex)}")
        except Exception as ex:
            raise InvokeError(f"Request failed: {str(ex)}")

    def _convert_messages_to_openai_format(self, messages: list[PromptMessage]) -> list[dict]:
        """
        Convert Dify messages to OpenAI chat format

        Args:
            messages: Dify prompt messages

        Returns:
            List of OpenAI-formatted messages
        """
        openai_messages = []
        
        for message in messages:
            if isinstance(message, SystemPromptMessage):
                openai_messages.append({
                    'role': 'system',
                    'content': message.content
                })
            elif isinstance(message, UserPromptMessage):
                if isinstance(message.content, str):
                    openai_messages.append({
                        'role': 'user',
                        'content': message.content
                    })
                else:
                    # Handle multimodal content
                    content_parts = []
                    for content_part in message.content:
                        if content_part.type == PromptMessageContentUnionTypes.TEXT:
                            content_parts.append({
                                'type': 'text',
                                'text': content_part.data
                            })
                        elif content_part.type == PromptMessageContentUnionTypes.IMAGE:
                            content_parts.append({
                                'type': 'image_url',
                                'image_url': {
                                    'url': content_part.data if isinstance(content_part.data, str) 
                                           else content_part.data.get('url', '')
                                }
                            })
                    openai_messages.append({
                        'role': 'user', 
                        'content': content_parts
                    })
            elif isinstance(message, AssistantPromptMessage):
                openai_message = {
                    'role': 'assistant',
                    'content': message.content or ''
                }
                
                # Add tool calls if present
                if message.tool_calls:
                    openai_message['tool_calls'] = [
                        {
                            'id': tool_call.id,
                            'type': 'function',
                            'function': {
                                'name': tool_call.function.name,
                                'arguments': tool_call.function.arguments
                            }
                        }
                        for tool_call in message.tool_calls
                    ]
                
                openai_messages.append(openai_message)
            elif isinstance(message, ToolPromptMessage):
                openai_messages.append({
                    'role': 'tool',
                    'tool_call_id': message.tool_call_id,
                    'content': str(message.content)
                })
        
        return openai_messages

    def _convert_tools_to_openai_format(self, tools: list[PromptMessageTool]) -> list[dict]:
        """
        Convert Dify tools to OpenAI function format

        Args:
            tools: Dify tools

        Returns:
            List of OpenAI-formatted tools
        """
        return [
            {
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.parameters
                }
            }
            for tool in tools
        ]

    def _get_chat_completions_url(self, base_url: str) -> str:
        """
        Get the chat completions endpoint URL

        Args:
            base_url: Base API URL

        Returns:
            Complete chat completions endpoint URL
        """
        return f"{base_url.rstrip('/')}/chat/completions"

    def _handle_non_streaming_response(
        self, 
        url: str, 
        headers: dict, 
        payload: dict, 
        model: str, 
        prompt_messages: list[PromptMessage]
    ) -> LLMResult:
        """
        Handle non-streaming API response

        Args:
            url: API endpoint URL
            headers: Request headers
            payload: Request payload
            model: Model name
            prompt_messages: Original prompt messages

        Returns:
            LLMResult with complete response
        """
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                self._handle_error_response(response)
            
            response_data = response.json()
            choice = response_data['choices'][0]
            message_data = choice['message']
            
            # Parse assistant message
            assistant_message = AssistantPromptMessage(
                content=message_data.get('content', '') or ''
            )
            
            # Handle tool calls if present
            if message_data.get('tool_calls'):
                tool_calls = []
                for tool_call in message_data['tool_calls']:
                    tool_calls.append(AssistantPromptMessage.ToolCall(
                        id=tool_call['id'],
                        type=tool_call['type'],
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name=tool_call['function']['name'],
                            arguments=tool_call['function']['arguments']
                        )
                    ))
                assistant_message.tool_calls = tool_calls
            
            # Parse usage information
            usage_data = response_data.get('usage', {})
            usage = LLMUsage(
                prompt_tokens=usage_data.get('prompt_tokens', 0),
                completion_tokens=usage_data.get('completion_tokens', 0),
                total_tokens=usage_data.get('total_tokens', 0)
            )
            
            return LLMResult(
                model=model,
                prompt_messages=prompt_messages,
                message=assistant_message,
                usage=usage,
                system_fingerprint=response_data.get('system_fingerprint', '')
            )

    def _handle_streaming_response(
        self, 
        url: str, 
        headers: dict, 
        payload: dict, 
        model: str, 
        prompt_messages: list[PromptMessage]
    ) -> Generator:
        """
        Handle streaming API response

        Args:
            url: API endpoint URL
            headers: Request headers
            payload: Request payload
            model: Model name
            prompt_messages: Original prompt messages

        Returns:
            Generator yielding LLMResultChunk objects
        """
        def create_stream():
            with httpx.stream('POST', url, headers=headers, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    self._handle_error_response(response)
                
                for line in response.iter_lines():
                    if not line or not line.startswith('data: '):
                        continue
                    
                    data = line[6:]  # Remove 'data: ' prefix
                    
                    if data.strip() == '[DONE]':
                        break
                    
                    try:
                        chunk_data = json.loads(data)
                        
                        if 'choices' not in chunk_data or not chunk_data['choices']:
                            continue
                        
                        choice = chunk_data['choices'][0]
                        delta = choice.get('delta', {})
                        
                        # Handle content delta
                        if delta.get('content'):
                            yield LLMResultChunk(
                                model=chunk_data.get('model', model),
                                prompt_messages=prompt_messages,
                                system_fingerprint=chunk_data.get('system_fingerprint', ''),
                                delta=LLMResultChunkDelta(
                                    index=choice.get('index', 0),
                                    message=AssistantPromptMessage(content=delta['content']),
                                    finish_reason=choice.get('finish_reason')
                                )
                            )
                        
                        # Handle tool call deltas
                        elif 'tool_calls' in delta:
                            tool_calls = []
                            for tool_call_delta in delta['tool_calls']:
                                tool_calls.append(AssistantPromptMessage.ToolCall(
                                    id=tool_call_delta.get('id', ''),
                                    type=tool_call_delta.get('type', 'function'),
                                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                        name=tool_call_delta.get('function', {}).get('name', ''),
                                        arguments=tool_call_delta.get('function', {}).get('arguments', '')
                                    )
                                ))
                            
                            yield LLMResultChunk(
                                model=chunk_data.get('model', model),
                                prompt_messages=prompt_messages,
                                system_fingerprint=chunk_data.get('system_fingerprint', ''),
                                delta=LLMResultChunkDelta(
                                    index=choice.get('index', 0),
                                    message=AssistantPromptMessage(content='', tool_calls=tool_calls),
                                    finish_reason=choice.get('finish_reason')
                                )
                            )
                        
                        # Handle final chunk with usage
                        elif choice.get('finish_reason'):
                            usage_data = chunk_data.get('usage', {})
                            usage = LLMUsage(
                                prompt_tokens=usage_data.get('prompt_tokens', 0),
                                completion_tokens=usage_data.get('completion_tokens', 0),
                                total_tokens=usage_data.get('total_tokens', 0)
                            ) if usage_data else None
                            
                            yield LLMResultChunk(
                                model=chunk_data.get('model', model),
                                prompt_messages=prompt_messages,
                                system_fingerprint=chunk_data.get('system_fingerprint', ''),
                                delta=LLMResultChunkDelta(
                                    index=choice.get('index', 0),
                                    message=AssistantPromptMessage(content=''),
                                    usage=usage,
                                    finish_reason=choice.get('finish_reason')
                                )
                            )
                    
                    except json.JSONDecodeError:
                        # Skip invalid JSON chunks
                        continue
        
        return create_stream()

    def _handle_error_response(self, response: httpx.Response) -> None:
        """
        Handle API error responses

        Args:
            response: HTTP response object

        Raises:
            Appropriate InvokeError subclass based on status code
        """
        try:
            error_data = response.json()
            error_message = error_data.get('error', {}).get('message', f'HTTP {response.status_code}')
        except:
            error_message = f'HTTP {response.status_code}'

        if response.status_code == 401:
            raise InvokeAuthorizationError(error_message)
        elif response.status_code == 400:
            raise InvokeBadRequestError(error_message)
        elif response.status_code == 429:
            raise InvokeRateLimitError(error_message)
        elif response.status_code >= 500:
            raise InvokeServerUnavailableError(error_message)
        else:
            raise InvokeError(error_message)