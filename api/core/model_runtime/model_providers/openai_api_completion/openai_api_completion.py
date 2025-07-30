"""
OpenAI API Completion Provider - specifically for /completions endpoint

This provider is designed to work exclusively with OpenAI-compatible text completion APIs.
It focuses on the /completions endpoint and provides optimized support for:
- Text completion
- Prompt-completion patterns
- Streaming responses
- Basic parameter controls
"""

import logging

from core.model_runtime.entities.model_entities import ModelType
from core.model_runtime.entities.provider_entities import (
    ConfigurateMethod,
)
from core.model_runtime.errors.validate import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class OpenAICompletionProvider:
    """
    OpenAI API Completion Provider - specifically for /completions endpoint

    This provider handles text completions using the OpenAI-compatible
    /completions API endpoint.
    """

    def __init__(self):
        self.provider_name = "openai_api_completion"
        self.supported_model_types = [ModelType.LLM]
        self.configurate_methods = [ConfigurateMethod.CUSTOMIZABLE_MODEL]

    def get_provider_schema(self) -> dict:
        """
        Returns the provider schema configuration

        This defines the provider's metadata, supported features,
        and credential requirements.
        """
        return {
            "provider": "openai_api_completion",
            "label": {"en_US": "OpenAI API Completion", "zh_Hans": "OpenAI API 补全"},
            "description": {
                "en_US": "OpenAI-compatible text completion API provider for /completions endpoint",
                "zh_Hans": "兼容 OpenAI 的文本补全 API 提供商，专用于 /completions 端点",
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
                        "default": "completion",
                        "options": [
                            {
                                "value": "completion",
                                "label": {
                                    "en_US": "Text Completions (/completions)",
                                    "zh_Hans": "文本补全 (/completions)",
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
            credentials: Provider credentials

        Raises:
            CredentialsValidateFailedError: If credentials are invalid
        """
        if not credentials:
            raise CredentialsValidateFailedError("Credentials cannot be empty")

        # Validate required fields
        required_fields = ["api_key", "endpoint_url"]
        for field in required_fields:
            if not credentials.get(field):
                raise CredentialsValidateFailedError(f"Missing required field: {field}")

        # Validate API key format (basic check)
        api_key = credentials.get("api_key", "")
        if not isinstance(api_key, str) or len(api_key.strip()) < 10:
            raise CredentialsValidateFailedError("API key must be a non-empty string with at least 10 characters")

        # Validate endpoint URL format
        endpoint_url = credentials.get("endpoint_url", "")
        if not isinstance(endpoint_url, str) or not endpoint_url.startswith(("http://", "https://")):
            raise CredentialsValidateFailedError("Endpoint URL must be a valid HTTP/HTTPS URL")

        # Validate mode
        mode = credentials.get("mode", "completion")
        if mode != "completion":
            raise CredentialsValidateFailedError("This provider only supports 'completion' mode")

    def validate_model_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        Args:
            model: Model name
            credentials: Model credentials

        Raises:
            CredentialsValidateFailedError: If credentials are invalid
        """
        if not model or not isinstance(model, str):
            raise CredentialsValidateFailedError("Model name must be a non-empty string")

        if not credentials:
            raise CredentialsValidateFailedError("Model credentials cannot be empty")

        # Validate context_size if provided
        context_size = credentials.get("context_size")
        if context_size is not None:
            try:
                context_size_int = int(context_size)
                if context_size_int <= 0 or context_size_int > 1000000:  # Reasonable upper limit
                    raise CredentialsValidateFailedError("Context size must be between 1 and 1,000,000")
            except (ValueError, TypeError):
                raise CredentialsValidateFailedError("Context size must be a valid integer")

        # Validate max_tokens if provided
        max_tokens = credentials.get("max_tokens")
        if max_tokens is not None:
            try:
                max_tokens_int = int(max_tokens)
                if max_tokens_int <= 0 or max_tokens_int > 100000:  # Reasonable upper limit
                    raise CredentialsValidateFailedError("Max tokens must be between 1 and 100,000")
            except (ValueError, TypeError):
                raise CredentialsValidateFailedError("Max tokens must be a valid integer")

    def get_supported_features(self) -> list[str]:
        """
        Get list of supported features for this provider

        Returns:
            List of supported feature names
        """
        return ["text_completion", "streaming", "prompt_completion", "single_turn", "basic_parameters"]

    def get_api_endpoint(self, base_url: str) -> str:
        """
        Get the specific API endpoint for this provider

        Args:
            base_url: Base API URL

        Returns:
            Complete endpoint URL for text completions
        """
        base_url = base_url.rstrip("/")
        return f"{base_url}/completions"

    def __str__(self) -> str:
        return f"OpenAICompletionProvider(provider={self.provider_name})"

    def __repr__(self) -> str:
        return self.__str__()
