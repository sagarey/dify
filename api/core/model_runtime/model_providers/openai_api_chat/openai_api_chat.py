"""
OpenAI API Chat Provider - specifically for /chat/completions endpoint

This provider is designed to work exclusively with OpenAI-compatible chat completion APIs.
It focuses on the /chat/completions endpoint and provides optimized support for:
- Multi-turn conversations
- Function calling
- System and user messages
- Streaming responses
"""

import logging

from core.model_runtime.entities.model_entities import ModelType
from core.model_runtime.entities.provider_entities import (
    ConfigurateMethod,
)
from core.model_runtime.errors.validate import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class OpenAIChatProvider:
    """
    OpenAI API Chat Provider - specifically for /chat/completions endpoint

    This provider handles chat-based completions using the OpenAI-compatible
    /chat/completions API endpoint.
    """

    def __init__(self):
        self.provider_name = "openai_api_chat"
        self.supported_model_types = [ModelType.LLM]
        self.configurate_methods = [ConfigurateMethod.CUSTOMIZABLE_MODEL]

    def get_provider_schema(self) -> dict:
        """
        Returns the provider schema configuration

        This defines the provider's metadata, supported features,
        and credential requirements.
        """
        return {
            "provider": "openai_api_chat",
            "label": {"en_US": "OpenAI API Chat", "zh_Hans": "OpenAI API 对话"},
            "description": {
                "en_US": "OpenAI-compatible chat completion API provider for /chat/completions endpoint",
                "zh_Hans": "兼容 OpenAI 的对话补全 API 提供商，专用于 /chat/completions 端点",
            },
            "icon_small": {"en_US": "icon_s_en.svg"},
            "icon_large": {"en_US": "icon_l_en.svg"},
            "supported_model_types": ["llm"],
            "configurate_methods": ["customizable-model"],
            "provider_credential_schema": {
                "credential_form_schemas": [
                    {
                        "variable": "api_key",
                        "label": {"en_US": "API Key", "zh_Hans": "API 密钥"},
                        "type": "secret-input",
                        "required": True,
                        "placeholder": {"en_US": "Enter your OpenAI API key", "zh_Hans": "输入您的 OpenAI API 密钥"},
                    },
                    {
                        "variable": "endpoint_url",
                        "label": {"en_US": "API Endpoint URL", "zh_Hans": "API 端点 URL"},
                        "type": "text-input",
                        "required": True,
                        "default": "https://api.openai.com/v1",
                        "placeholder": {"en_US": "Enter your API endpoint URL", "zh_Hans": "输入您的 API 端点 URL"},
                    },
                    {
                        "variable": "mode",
                        "label": {"en_US": "API Mode", "zh_Hans": "API 模式"},
                        "type": "select",
                        "required": True,
                        "default": "chat",
                        "options": [
                            {
                                "value": "chat",
                                "label": {
                                    "en_US": "Chat Completions (/chat/completions)",
                                    "zh_Hans": "对话补全 (/chat/completions)",
                                },
                            }
                        ],
                    },
                ]
            },
            "model_credential_schema": {
                "model": {
                    "label": {"en_US": "Model Name", "zh_Hans": "模型名称"},
                    "placeholder": {"en_US": "Enter your model name", "zh_Hans": "输入模型名称"},
                },
                "credential_form_schemas": [
                    {
                        "variable": "context_size",
                        "label": {"en_US": "Context Size", "zh_Hans": "上下文长度"},
                        "type": "text-input",
                        "required": False,
                        "default": "4096",
                        "placeholder": {
                            "en_US": "Maximum context length for the model",
                            "zh_Hans": "模型的最大上下文长度",
                        },
                    },
                    {
                        "variable": "max_tokens",
                        "label": {"en_US": "Max Tokens", "zh_Hans": "最大输出长度"},
                        "type": "text-input",
                        "required": False,
                        "default": "2048",
                        "placeholder": {
                            "en_US": "Maximum number of tokens to generate",
                            "zh_Hans": "生成的最大 token 数量",
                        },
                    },
                ],
            },
        }

    def validate_provider_credentials(self, credentials: dict) -> None:
        """
        Validate provider credentials

        Args:
            credentials: Provider credentials dictionary

        Raises:
            CredentialsValidateFailedError: If credentials are invalid
        """
        if not credentials.get("api_key"):
            raise CredentialsValidateFailedError("API Key is required")

        if not credentials.get("endpoint_url"):
            raise CredentialsValidateFailedError("API Endpoint URL is required")

        # Ensure this provider is configured for chat mode
        mode = credentials.get("mode", "chat")
        if mode != "chat":
            raise CredentialsValidateFailedError("This provider only supports chat mode (/chat/completions endpoint)")

    def validate_model_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model-specific credentials

        Args:
            model: Model name
            credentials: Model credentials dictionary

        Raises:
            CredentialsValidateFailedError: If model credentials are invalid
        """
        if not model or not model.strip():
            raise CredentialsValidateFailedError("Model name is required")

        # Validate context size if provided
        context_size = credentials.get("context_size")
        if context_size:
            try:
                context_size_int = int(context_size)
                if context_size_int <= 0:
                    raise CredentialsValidateFailedError("Context size must be a positive integer")
            except ValueError:
                raise CredentialsValidateFailedError("Context size must be a valid integer")

        # Validate max tokens if provided
        max_tokens = credentials.get("max_tokens")
        if max_tokens:
            try:
                max_tokens_int = int(max_tokens)
                if max_tokens_int <= 0:
                    raise CredentialsValidateFailedError("Max tokens must be a positive integer")
            except ValueError:
                raise CredentialsValidateFailedError("Max tokens must be a valid integer")

    def get_supported_features(self) -> list[str]:
        """
        Get list of supported features for this provider

        Returns:
            List of supported feature names
        """
        return [
            "chat_completions",
            "streaming",
            "function_calling",
            "multi_turn_conversation",
            "system_messages",
            "user_messages",
            "assistant_messages",
        ]

    def get_api_endpoint(self, base_url: str) -> str:
        """
        Get the specific API endpoint for this provider

        Args:
            base_url: Base API URL

        Returns:
            Complete endpoint URL for chat completions
        """
        base_url = base_url.rstrip("/")
        return f"{base_url}/chat/completions"

    def __str__(self) -> str:
        return f"OpenAIChatProvider(provider={self.provider_name})"

    def __repr__(self) -> str:
        return self.__str__()
