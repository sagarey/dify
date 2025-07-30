"""
OpenAI API Completion Large Language Model Implementation

This module implements the LLM interface for OpenAI-compatible text completion APIs.
It specifically targets the /completions endpoint and provides:
- Streaming and non-streaming completions
- Text completion and generation
- Prompt-based interactions
- Basic parameter controls
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
    SystemPromptMessage,
    UserPromptMessage,
)
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


class OpenAICompletionLargeLanguageModel(LargeLanguageModel):
    """
    OpenAI API Completion Large Language Model - specifically for /completions endpoint
    
    This implementation focuses exclusively on text completions,
    providing optimized support for prompt-completion style interactions.
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: Optional[dict] = None,
        tools: Optional[list] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model for text completions

        Args:
            model: Model name
            credentials: Model credentials
            prompt_messages: Prompt messages (converted to single prompt)
            model_parameters: Model parameters
            tools: Tools (not supported for completions)
            stop: Stop words
            stream: Whether to stream response
            user: Unique user identifier

        Returns:
            LLMResult for non-streaming, Generator for streaming
        """
        # Validate that we're using completion mode
        if credentials.get('mode', 'completion') != 'completion':
            raise CredentialsValidateFailedError(
                'This provider only supports completion mode'
            )

        # Note: tools are not supported in completion mode
        if tools:
            logger.warning("Tools/function calling is not supported in completion mode, ignoring")

        return self._text_completion_request(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters or {},
            stop=stop,
            stream=stream,
            user=user,
        )

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages

        Args:
            model: Model name
            credentials: Model credentials
            prompt_messages: Prompt messages
            tools: Tools (ignored for completions)

        Returns:
            Estimated number of tokens
        """
        prompt_text = self._convert_messages_to_prompt(prompt_messages)
        
        # Simple token estimation - in production, use proper tokenizer
        # Rough estimation: 1 token ≈ 4 characters
        return max(len(prompt_text) // 4, 1)

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
            self._text_completion_request(
                model=model,
                credentials=credentials,
                prompt_messages=[UserPromptMessage(content="Hello")],
                model_parameters={"max_tokens": 5},
                stream=False,
            )
        except Exception as ex:
            raise CredentialsValidateFailedError(f"Credential validation failed: {str(ex)}")

    def _text_completion_request(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Make text completion request to OpenAI-compatible API

        Args:
            model: Model name
            credentials: API credentials
            prompt_messages: Prompt messages (converted to single prompt)
            model_parameters: Model parameters
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

        # Convert Dify messages to single prompt text
        prompt = self._convert_messages_to_prompt(prompt_messages)

        # Prepare request payload
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': stream,
        }

        # Add model parameters
        payload.update(model_parameters)

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
            estimated_prompt_tokens = self.get_num_tokens(model, credentials, prompt_messages)
            max_response_tokens = min(
                int(credentials.get('max_tokens', 2048)),
                max(context_size - estimated_prompt_tokens - 100, 100)
            )
            payload['max_tokens'] = max_response_tokens

        # Make API request
        api_url = self._get_completions_url(endpoint_url)
        
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

    def _convert_messages_to_prompt(self, messages: list[PromptMessage]) -> str:
        """
        Convert Dify messages to single prompt text for completion API

        Args:
            messages: Dify prompt messages

        Returns:
            Single prompt text
        """
        prompt_parts = []
        
        for message in messages:
            if isinstance(message, SystemPromptMessage):
                # Add system message as prefix
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, UserPromptMessage):
                if isinstance(message.content, str):
                    prompt_parts.append(message.content)
                else:
                    # Extract text from multimodal content
                    text_parts = []
                    for content_part in message.content:
                        if hasattr(content_part, 'data') and isinstance(content_part.data, str):
                            text_parts.append(content_part.data)
                    prompt_parts.append(' '.join(text_parts))
            elif isinstance(message, AssistantPromptMessage):
                # Add assistant message as context
                if message.content:
                    prompt_parts.append(message.content)
        
        return '\n'.join(prompt_parts)

    def _get_completions_url(self, base_url: str) -> str:
        """
        Get the completions endpoint URL

        Args:
            base_url: Base API URL

        Returns:
            Complete completions endpoint URL
        """
        return f"{base_url.rstrip('/')}/completions"

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
            
            # Parse completion text
            completion_text = choice.get('text', '')
            
            assistant_message = AssistantPromptMessage(content=completion_text)
            
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
                        
                        # Handle text delta
                        if 'text' in choice:
                            text_delta = choice['text']
                            
                            if text_delta:  # Only yield if there's actual content
                                yield LLMResultChunk(
                                    model=chunk_data.get('model', model),
                                    prompt_messages=prompt_messages,
                                    system_fingerprint=chunk_data.get('system_fingerprint', ''),
                                    delta=LLMResultChunkDelta(
                                        index=choice.get('index', 0),
                                        message=AssistantPromptMessage(content=text_delta),
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