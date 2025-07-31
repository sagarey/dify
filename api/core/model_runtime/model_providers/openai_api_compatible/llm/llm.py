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

from core.model_runtime.entities.llm_entities import LLMResult
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

    def _get_chat_llm(self):
        """Get chat LLM instance with lazy loading to avoid circular imports"""
        try:
            from core.model_runtime.model_providers.openai_api_chat.llm.llm import (
                OpenAIChatLargeLanguageModel,
            )

            return OpenAIChatLargeLanguageModel()
        except ImportError as e:
            logger.exception("Failed to import OpenAIChatLargeLanguageModel")
            raise InvokeError("Chat provider not available") from e

    def _get_completion_llm(self):
        """Get completion LLM instance with lazy loading to avoid circular imports"""
        try:
            from core.model_runtime.model_providers.openai_api_completion.llm.llm import (
                OpenAICompletionLargeLanguageModel,
            )

            return OpenAICompletionLargeLanguageModel()
        except ImportError as e:
            logger.exception("Failed to import OpenAICompletionLargeLanguageModel")
            raise InvokeError("Completion provider not available") from e

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
            stacklevel=2,
        )

        logger.warning(
            "DEPRECATED: Invoking model '%s' via deprecated openai_api_compatible LLM. "
            "Please migrate to openai_api_chat or openai_api_completion.",
            model,
        )

        # Route to appropriate provider based on mode
        mode = credentials.get("mode", "chat")

        try:
            if mode == "chat":
                # Route to chat provider
                chat_llm = self._get_chat_llm()
                return chat_llm._invoke(
                    model=model,
                    credentials=credentials,
                    prompt_messages=prompt_messages,
                    model_parameters=model_parameters,
                    tools=tools,
                    stop=stop,
                    stream=stream,
                    user=user,
                )
            else:
                # Route to completion provider
                completion_llm = self._get_completion_llm()
                return completion_llm._invoke(
                    model=model,
                    credentials=credentials,
                    prompt_messages=prompt_messages,
                    model_parameters=model_parameters,
                    tools=tools,
                    stop=stop,
                    stream=stream,
                    user=user,
                )
        except Exception as e:
            logger.exception("Failed to route request to appropriate provider")
            raise InvokeError(f"Request routing failed: {e}") from e

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens with automatic routing
        """
        warnings.warn(
            "The 'openai_api_compatible' LLM is deprecated. "
            "Please migrate to 'openai_api_chat' or 'openai_api_completion'.",
            DeprecationWarning,
            stacklevel=2,
        )

        mode = credentials.get("mode", "chat")

        try:
            if mode == "chat":
                chat_llm = self._get_chat_llm()
                return chat_llm.get_num_tokens(
                    model=model,
                    credentials=credentials,
                    prompt_messages=prompt_messages,
                    tools=tools,
                )
            else:
                completion_llm = self._get_completion_llm()
                return completion_llm.get_num_tokens(
                    model=model,
                    credentials=credentials,
                    prompt_messages=prompt_messages,
                    tools=tools,
                )
        except Exception as e:
            logger.exception("Failed to get token count via routed provider")
            raise InvokeError(f"Token counting failed: {e}") from e

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate credentials with automatic routing
        """
        warnings.warn(
            "The 'openai_api_compatible' LLM is deprecated. "
            "Please migrate to 'openai_api_chat' or 'openai_api_completion'.",
            DeprecationWarning,
            stacklevel=2,
        )

        mode = credentials.get("mode", "chat")

        try:
            if mode == "chat":
                chat_llm = self._get_chat_llm()
                chat_llm.validate_credentials(model, credentials)
            else:
                completion_llm = self._get_completion_llm()
                completion_llm.validate_credentials(model, credentials)
        except Exception as e:
            logger.exception("Failed to validate credentials via routed provider")
            raise InvokeError(f"Credential validation failed: {e}") from e
