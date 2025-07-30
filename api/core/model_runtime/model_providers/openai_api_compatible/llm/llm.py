"""
OpenAI API Compatible LLM - Legacy Compatibility Layer

This LLM implementation maintains backward compatibility for existing users.
It automatically routes requests to the appropriate new providers based on configuration.

DEPRECATED: This LLM is deprecated. Please migrate to:
- openai_api_chat for chat-based completions
- openai_api_completion for text completions
"""

import logging
import warnings
from collections.abc import Generator
from typing import Optional, Union

from core.model_runtime.entities.llm_entities import LLMResult, LLMResultChunk, LLMUsage
from core.model_runtime.entities.message_entities import PromptMessage, PromptMessageTool
from core.model_runtime.errors.invoke import InvokeError
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel

logger = logging.getLogger(__name__)


class OpenAICompatibleLargeLanguageModel(LargeLanguageModel):
    """
    OpenAI API Compatible LLM - Legacy Compatibility Layer
    
    DEPRECATED: This LLM is deprecated and will be removed in a future version.
    Routes requests to the appropriate specialized provider based on mode configuration.
    """

    def _invoke(
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
        Invoke LLM with automatic routing to new providers
        """
        # Issue deprecation warning
        warnings.warn(
            "The 'openai_api_compatible' LLM is deprecated and will be removed in a future version. "
            "Please migrate to 'openai_api_chat' for chat completions or 'openai_api_completion' for text completions.",
            DeprecationWarning,
            stacklevel=2
        )
        
        logger.warning(
            "DEPRECATED: Invoking model '%s' via deprecated openai_api_compatible LLM. "
            "Please migrate to openai_api_chat or openai_api_completion.",
            model
        )

        # Route to appropriate provider based on mode
        mode = credentials.get('mode', 'chat')
        
        try:
            if mode == 'chat':
                # Route to chat provider
                from core.model_runtime.model_providers.openai_api_chat.llm.llm import OpenAIChatLargeLanguageModel
                chat_llm = OpenAIChatLargeLanguageModel()
                return chat_llm._invoke(
                    model=model,
                    credentials=credentials,
                    prompt_messages=prompt_messages,
                    model_parameters=model_parameters,
                    tools=tools,
                    stop=stop,
                    stream=stream,
                    user=user
                )
            else:
                # Route to completion provider
                from core.model_runtime.model_providers.openai_api_completion.llm.llm import OpenAICompletionLargeLanguageModel
                completion_llm = OpenAICompletionLargeLanguageModel()
                return completion_llm._invoke(
                    model=model,
                    credentials=credentials,
                    prompt_messages=prompt_messages,
                    model_parameters=model_parameters,
                    tools=tools,
                    stop=stop,
                    stream=stream,
                    user=user
                )
        except Exception as e:
            logger.error("Failed to route request to appropriate provider: %s", e)
            raise InvokeError(f"Request routing failed: {e}")

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens with routing to appropriate provider
        """
        # Issue deprecation warning
        warnings.warn(
            "The 'openai_api_compatible' LLM is deprecated. "
            "Please migrate to 'openai_api_chat' or 'openai_api_completion'.",
            DeprecationWarning,
            stacklevel=2
        )

        mode = credentials.get('mode', 'chat')
        
        try:
            if mode == 'chat':
                from core.model_runtime.model_providers.openai_api_chat.llm.llm import OpenAIChatLargeLanguageModel
                chat_llm = OpenAIChatLargeLanguageModel()
                return chat_llm.get_num_tokens(model, credentials, prompt_messages, tools)
            else:
                from core.model_runtime.model_providers.openai_api_completion.llm.llm import OpenAICompletionLargeLanguageModel
                completion_llm = OpenAICompletionLargeLanguageModel()
                return completion_llm.get_num_tokens(model, credentials, prompt_messages, tools)
        except Exception as e:
            logger.error("Failed to get token count via routed provider: %s", e)
            # Return a reasonable default if routing fails
            total_tokens = 0
            for message in prompt_messages:
                if hasattr(message, 'content') and message.content:
                    total_tokens += len(str(message.content)) // 4
            return total_tokens

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate credentials with routing to appropriate provider
        """
        # Issue deprecation warning
        warnings.warn(
            "The 'openai_api_compatible' LLM is deprecated. "
            "Please migrate to 'openai_api_chat' or 'openai_api_completion'.",
            DeprecationWarning,
            stacklevel=2
        )
        
        logger.warning(
            "DEPRECATED: Validating credentials for model '%s' via deprecated openai_api_compatible LLM. "
            "Please migrate to openai_api_chat or openai_api_completion.",
            model
        )

        mode = credentials.get('mode', 'chat')
        
        try:
            if mode == 'chat':
                from core.model_runtime.model_providers.openai_api_chat.llm.llm import OpenAIChatLargeLanguageModel
                chat_llm = OpenAIChatLargeLanguageModel()
                chat_llm.validate_credentials(model, credentials)
            else:
                from core.model_runtime.model_providers.openai_api_completion.llm.llm import OpenAICompletionLargeLanguageModel
                completion_llm = OpenAICompletionLargeLanguageModel()
                completion_llm.validate_credentials(model, credentials)
        except Exception as e:
            logger.error("Failed to validate credentials via routed provider: %s", e)
            raise