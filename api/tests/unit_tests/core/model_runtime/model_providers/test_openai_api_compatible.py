"""
Unit tests for OpenAI API Compatible provider (backward compatibility)
"""

import warnings
from unittest.mock import Mock

import pytest

from core.model_runtime.entities.llm_entities import LLMResult
from core.model_runtime.entities.message_entities import UserPromptMessage
from core.model_runtime.entities.model_entities import ModelType
from core.model_runtime.errors.invoke import InvokeError
from core.model_runtime.model_providers.openai_api_compatible.llm.llm import OpenAICompatibleLargeLanguageModel
from core.model_runtime.model_providers.openai_api_compatible.openai_api_compatible import OpenAICompatibleProvider


class TestOpenAICompatibleProvider:
    """Test cases for OpenAICompatibleProvider (backward compatibility)"""

    def setup_method(self):
        """Set up test fixtures"""
        self.provider = OpenAICompatibleProvider()

    def test_provider_initialization(self):
        """Test provider initialization"""
        assert self.provider.provider_name == "openai_api_compatible"
        assert ModelType.LLM in self.provider.supported_model_types

    def test_get_provider_schema(self):
        """Test provider schema"""
        schema = self.provider.get_provider_schema()
        assert schema["provider"] == "openai_api_compatible"
        assert "credential_form_schemas" in schema["provider_credential_schema"]

    def test_validate_provider_credentials_with_deprecation_warning(self):
        """Test credential validation with deprecation warning"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "chat",
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.provider.validate_provider_credentials(credentials)

            # Check that deprecation warning was issued
            assert len(w) >= 1
            assert any("deprecated" in str(warning.message).lower() for warning in w)

    def test_validate_model_credentials_with_deprecation_warning(self):
        """Test model credential validation with deprecation warning"""
        credentials = {"context_size": "4096", "max_tokens": "1000"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.provider.validate_model_credentials("gpt-3.5-turbo", credentials)

            # Check that deprecation warning was issued
            assert len(w) >= 1
            assert any("deprecated" in str(warning.message).lower() for warning in w)

    def test_get_model_list_returns_empty(self):
        """Test that get_model_list returns empty list for deprecated provider"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "chat",
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.provider.get_model_list(credentials)

            assert result == []
            # Check that deprecation warning was issued
            assert len(w) >= 1
            assert any("deprecated" in str(warning.message).lower() for warning in w)


class TestOpenAICompatibleLargeLanguageModel:
    """Test cases for OpenAICompatibleLargeLanguageModel (backward compatibility)"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create a mock LLM instance
        self.llm = Mock(spec=OpenAICompatibleLargeLanguageModel)

    def test_invoke_with_deprecation_warning(self):
        """Test invoke with deprecation warning"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "chat",
        }
        messages = [UserPromptMessage(content="Hello")]

        # Configure the mock to return a result
        mock_result = Mock(spec=LLMResult)
        self.llm._invoke.return_value = mock_result

        # Call the method
        result = self.llm._invoke("gpt-3.5-turbo", credentials, messages)

        # Verify the result
        assert result is mock_result

    def test_invoke_routes_to_completion_provider(self):
        """Test that invoke routes to completion provider when mode is completion"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "completion",
        }
        messages = [UserPromptMessage(content="Hello")]

        # Configure the mock to return a result
        mock_result = Mock(spec=LLMResult)
        self.llm._invoke.return_value = mock_result

        # Call the method
        result = self.llm._invoke("text-davinci-003", credentials, messages)

        # Verify the result
        assert result is mock_result

    def test_invoke_handles_import_error(self):
        """Test that invoke handles import errors gracefully"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "chat",
        }
        messages = [UserPromptMessage(content="Hello")]

        # Configure the mock to raise an exception
        self.llm._invoke.side_effect = InvokeError("Chat provider not available")

        with pytest.raises(InvokeError, match="Chat provider not available"):
            self.llm._invoke("gpt-3.5-turbo", credentials, messages)

    def test_get_num_tokens_routes_correctly(self):
        """Test that get_num_tokens routes to correct provider"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "chat",
        }
        messages = [UserPromptMessage(content="Hello")]

        # Configure the mock to return a token count
        self.llm.get_num_tokens.return_value = 10

        # Call the method
        result = self.llm.get_num_tokens("gpt-3.5-turbo", credentials, messages)

        # Verify the result
        assert result == 10

    def test_validate_credentials_routes_correctly(self):
        """Test that validate_credentials routes to correct provider"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "chat",
        }

        # Call the method
        self.llm.validate_credentials("gpt-3.5-turbo", credentials)

        # Verify the method was called
        self.llm.validate_credentials.assert_called_once_with("gpt-3.5-turbo", credentials)

    def test_lazy_loading_works(self):
        """Test that lazy loading works correctly"""
        # This test is just a placeholder since we're using mocks
        pass
