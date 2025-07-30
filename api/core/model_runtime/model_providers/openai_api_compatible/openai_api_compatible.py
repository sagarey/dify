"""
OpenAI API Compatible Provider - Legacy Compatibility Layer

This provider maintains backward compatibility for existing users of openai_api_compatible.
It automatically routes requests to the new specialized providers:
- openai_api_chat: for /chat/completions endpoint
- openai_api_completion: for /completions endpoint

DEPRECATED: This provider is deprecated. Please migrate to:
- openai_api_chat for chat-based completions
- openai_api_completion for text completions
"""

import logging
import warnings
from typing import Optional

from core.model_runtime.entities.model_entities import ModelType
from core.model_runtime.entities.provider_entities import (
    ConfigurateMethod,
    FormType,
    ProviderEntity,
    ProviderQuotaType,
    QuotaUnit,
    SystemConfigurationStatus,
)
from core.model_runtime.errors.validate import CredentialsValidateFailedError
from core.model_runtime.model_providers.__base.model_provider import ModelProvider

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(ModelProvider):
    """
    OpenAI API Compatible Provider - Legacy Compatibility Layer
    
    DEPRECATED: This provider is deprecated and will be removed in a future version.
    Please migrate to openai_api_chat or openai_api_completion providers.
    """

    def get_provider_schema(self) -> ProviderEntity:
        """
        Returns the provider schema with deprecation warning
        """
        return ProviderEntity(
            provider="openai_api_compatible",
            label=ProviderEntity.ProviderLabelEntity(
                en_US="OpenAI API Compatible (Deprecated)",
                zh_Hans="OpenAI API 兼容 (已弃用)"
            ),
            description=ProviderEntity.ProviderDescriptionEntity(
                en_US="DEPRECATED: Please use openai_api_chat or openai_api_completion instead. "
                      "Generic provider for OpenAI-compatible APIs supporting both chat and completion endpoints.",
                zh_Hans="已弃用：请使用 openai_api_chat 或 openai_api_completion。"
                        "支持聊天和补全端点的通用 OpenAI 兼容 API 提供商。"
            ),
            background="#1C3A32",
            icon_small=ProviderEntity.ProviderIconEntity(
                en_US=self.get_icon(),
                zh_Hans=self.get_icon()
            ),
            icon_large=ProviderEntity.ProviderIconEntity(
                en_US=self.get_icon(),
                zh_Hans=self.get_icon()
            ),
            supported_model_types=[ModelType.LLM],
            configurate_methods=[ConfigurateMethod.CUSTOMIZABLE_MODEL],
            provider_credential_schema=ProviderEntity.ProviderCredentialSchema(
                credential_form_schemas=[
                    ProviderEntity.CredentialFormSchema(
                        variable="api_key",
                        label=ProviderEntity.CredentialFormSchema.CredentialFormSchemaLabel(
                            en_US="API Key",
                            zh_Hans="API Key"
                        ),
                        type=FormType.SECRET_INPUT,
                        required=True,
                        placeholder=ProviderEntity.CredentialFormSchema.CredentialFormSchemaPlaceholder(
                            en_US="Enter your API Key",
                            zh_Hans="输入你的 API Key"
                        )
                    ),
                    ProviderEntity.CredentialFormSchema(
                        variable="api_base",
                        label=ProviderEntity.CredentialFormSchema.CredentialFormSchemaLabel(
                            en_US="API Base",
                            zh_Hans="API Base"
                        ),
                        type=FormType.TEXT_INPUT,
                        required=True,
                        placeholder=ProviderEntity.CredentialFormSchema.CredentialFormSchemaPlaceholder(
                            en_US="Enter your API Base URL",
                            zh_Hans="输入你的 API Base URL"
                        )
                    ),
                    ProviderEntity.CredentialFormSchema(
                        variable="mode",
                        label=ProviderEntity.CredentialFormSchema.CredentialFormSchemaLabel(
                            en_US="Completion Mode (DEPRECATED - use dedicated providers instead)",
                            zh_Hans="补全模式 (已弃用 - 请使用专用提供商)"
                        ),
                        type=FormType.SELECT,
                        required=False,
                        default="chat",
                        options=[
                            ProviderEntity.CredentialFormSchema.CredentialFormSchemaOption(
                                label=ProviderEntity.CredentialFormSchema.CredentialFormSchemaLabel(
                                    en_US="Chat Completions (/chat/completions) - Use openai_api_chat instead",
                                    zh_Hans="聊天补全 (/chat/completions) - 请使用 openai_api_chat"
                                ),
                                value="chat"
                            ),
                            ProviderEntity.CredentialFormSchema.CredentialFormSchemaOption(
                                label=ProviderEntity.CredentialFormSchema.CredentialFormSchemaLabel(
                                    en_US="Text Completions (/completions) - Use openai_api_completion instead",
                                    zh_Hans="文本补全 (/completions) - 请使用 openai_api_completion"
                                ),
                                value="completion"
                            )
                        ]
                    )
                ]
            ),
            model_credential_schema=ProviderEntity.ModelCredentialSchema(
                model={
                    "label": ProviderEntity.ModelCredentialSchema.ModelCredentialSchemaLabel(
                        en_US="Model Name",
                        zh_Hans="模型名称"
                    ),
                    "placeholder": ProviderEntity.ModelCredentialSchema.ModelCredentialSchemaPlaceholder(
                        en_US="Enter your model name",
                        zh_Hans="输入模型名称"
                    )
                },
                credential_form_schemas=[
                    ProviderEntity.CredentialFormSchema(
                        variable="context_size",
                        label=ProviderEntity.CredentialFormSchema.CredentialFormSchemaLabel(
                            en_US="Context Size",
                            zh_Hans="上下文长度"
                        ),
                        type=FormType.TEXT_INPUT,
                        required=True,
                        default="4096",
                        placeholder=ProviderEntity.CredentialFormSchema.CredentialFormSchemaPlaceholder(
                            en_US="Enter context size",
                            zh_Hans="请输入上下文长度"
                        )
                    ),
                    ProviderEntity.CredentialFormSchema(
                        variable="max_tokens",
                        label=ProviderEntity.CredentialFormSchema.CredentialFormSchemaLabel(
                            en_US="Max Tokens",
                            zh_Hans="最大 token 数"
                        ),
                        type=FormType.TEXT_INPUT,
                        required=True,
                        default="512",
                        placeholder=ProviderEntity.CredentialFormSchema.CredentialFormSchemaPlaceholder(
                            en_US="Enter max tokens",
                            zh_Hans="请输入最大 token 数"
                        )
                    )
                ]
            )
        )

    def validate_provider_credentials(self, credentials: dict) -> None:
        """
        Validate provider credentials with deprecation warning
        """
        # Issue deprecation warning
        warnings.warn(
            "The 'openai_api_compatible' provider is deprecated and will be removed in a future version. "
            "Please migrate to 'openai_api_chat' for chat completions or 'openai_api_completion' for text completions.",
            DeprecationWarning,
            stacklevel=2
        )
        
        logger.warning(
            "DEPRECATED: openai_api_compatible provider is deprecated. "
            "Please migrate to openai_api_chat or openai_api_completion."
        )

        # Route validation to appropriate provider
        try:
            mode = credentials.get('mode', 'chat')
            
            if mode == 'chat':
                # Import and use chat provider for validation
                from core.model_runtime.model_providers.openai_api_chat.openai_api_chat import OpenAIChatProvider
                chat_provider = OpenAIChatProvider()
                chat_provider.validate_provider_credentials(credentials)
            else:
                # Import and use completion provider for validation
                from core.model_runtime.model_providers.openai_api_completion.openai_api_completion import OpenAICompletionProvider
                completion_provider = OpenAICompletionProvider()
                completion_provider.validate_provider_credentials(credentials)
                
        except Exception as e:
            logger.error(f"Failed to validate credentials via routed provider: {e}")
            raise CredentialsValidateFailedError(f"Credential validation failed: {e}")

    def validate_model_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials with deprecation warning
        """
        # Issue deprecation warning
        warnings.warn(
            "The 'openai_api_compatible' provider is deprecated. "
            "Please migrate to 'openai_api_chat' or 'openai_api_completion'.",
            DeprecationWarning,
            stacklevel=2
        )
        
        logger.warning(
            f"DEPRECATED: Validating model '{model}' via deprecated openai_api_compatible provider. "
            "Please migrate to openai_api_chat or openai_api_completion."
        )

        # Route validation to appropriate provider
        try:
            mode = credentials.get('mode', 'chat')
            
            if mode == 'chat':
                from core.model_runtime.model_providers.openai_api_chat.openai_api_chat import OpenAIChatProvider
                chat_provider = OpenAIChatProvider()
                chat_provider.validate_model_credentials(model, credentials)
            else:
                from core.model_runtime.model_providers.openai_api_completion.openai_api_completion import OpenAICompletionProvider
                completion_provider = OpenAICompletionProvider()
                completion_provider.validate_model_credentials(model, credentials)
                
        except Exception as e:
            logger.error(f"Failed to validate model credentials via routed provider: {e}")
            raise CredentialsValidateFailedError(f"Model credential validation failed: {e}")

    def get_model_list(self, credentials: dict) -> list:
        """
        Get model list with deprecation warning
        """
        warnings.warn(
            "The 'openai_api_compatible' provider is deprecated. "
            "Please migrate to 'openai_api_chat' or 'openai_api_completion'.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Return empty list for deprecated provider
        return []

    @staticmethod
    def get_icon() -> str:
        return "icon_data_here"  # Placeholder for icon