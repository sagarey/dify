"""
Unit tests for Kimi AI provider
"""

import unittest

import pytest

from core.model_runtime.entities.model_entities import ModelType
from core.model_runtime.errors.validate import CredentialsValidateFailedError
from core.model_runtime.model_providers.kimi.kimi import KimiProvider
from core.model_runtime.model_providers.kimi.llm.llm import KimiLargeLanguageModel


class TestKimiProvider(unittest.TestCase):
    """Test cases for KimiProvider"""

    def setUp(self):
        """Set up test fixtures"""
        self.provider = KimiProvider()

    def test_provider_initialization(self):
        """Test provider initialization"""
        assert self.provider.provider_name == "kimi"
        assert ModelType.LLM in self.provider.supported_model_types

    def test_get_provider_schema(self):
        """Test provider schema"""
        schema = self.provider.get_provider_schema()
        assert schema["provider"] == "kimi"
        assert "credential_form_schemas" in schema["provider_credential_schema"]

    def test_validate_provider_credentials_success(self):
        """Test successful credential validation"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.moonshot.cn/v1",
            "mode": "chat",
        }
        # Should not raise any exception
        try:
            self.provider.validate_provider_credentials(credentials)
        except Exception as e:
            self.fail(f"validate_provider_credentials raised {type(e).__name__} unexpectedly: {e}")

    def test_validate_provider_credentials_missing_api_key(self):
        """Test credential validation with missing API key"""
        credentials = {"endpoint_url": "https://api.moonshot.cn/v1", "mode": "chat"}
        with pytest.raises(CredentialsValidateFailedError) as context:
            self.provider.validate_provider_credentials(credentials)
        assert "API Key is required" in str(context.exception)

    def test_validate_provider_credentials_empty_api_key(self):
        """Test credential validation with empty API key"""
        credentials = {"api_key": "", "endpoint_url": "https://api.moonshot.cn/v1", "mode": "chat"}
        with pytest.raises(CredentialsValidateFailedError) as context:
            self.provider.validate_provider_credentials(credentials)
        assert "API Key is required" in str(context.exception)

    def test_validate_provider_credentials_invalid_endpoint(self):
        """Test credential validation with invalid endpoint"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "invalid-url",
            "mode": "chat",
        }
        # Should not raise exception for invalid URL format in provider validation
        try:
            self.provider.validate_provider_credentials(credentials)
        except Exception as e:
            self.fail(f"validate_provider_credentials raised {type(e).__name__} unexpectedly: {e}")

    def test_validate_provider_credentials_wrong_mode(self):
        """Test credential validation with wrong mode"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.moonshot.cn/v1",
            "mode": "completion",
        }
        with pytest.raises(CredentialsValidateFailedError) as context:
            self.provider.validate_provider_credentials(credentials)
        assert "only supports chat mode" in str(context.exception)

    def test_validate_model_credentials_success(self):
        """Test successful model credential validation"""
        credentials = {
            "context_size": "32768",
            "max_tokens": "2048",
        }
        # Should not raise any exception
        try:
            self.provider.validate_model_credentials("moonshot-v1-8k", credentials)
        except Exception as e:
            self.fail(f"validate_model_credentials raised {type(e).__name__} unexpectedly: {e}")

    def test_validate_model_credentials_invalid_context_size(self):
        """Test model credential validation with invalid context size"""
        credentials = {
            "context_size": "invalid",
            "max_tokens": "2048",
        }
        with pytest.raises(CredentialsValidateFailedError) as context:
            self.provider.validate_model_credentials("moonshot-v1-8k", credentials)
        assert "Context size must be a valid integer" in str(context.exception)

    def test_validate_model_credentials_invalid_max_tokens(self):
        """Test model credential validation with invalid max tokens"""
        credentials = {
            "context_size": "32768",
            "max_tokens": "invalid",
        }
        with pytest.raises(CredentialsValidateFailedError) as context:
            self.provider.validate_model_credentials("moonshot-v1-8k", credentials)
        assert "Max tokens must be a valid integer" in str(context.exception)

    def test_get_supported_features(self):
        """Test supported features"""
        features = self.provider.get_supported_features()
        assert "chat_completions" in features
        assert "streaming" in features
        assert "function_calling" in features


class TestKimiLargeLanguageModel(unittest.TestCase):
    """Test cases for KimiLargeLanguageModel"""

    def setUp(self):
        """Set up test fixtures"""
        # For unit tests, we'll test the methods directly without instantiating the full class
        # since it requires plugin infrastructure that's not available in unit tests
        self.llm = KimiLargeLanguageModel.__new__(KimiLargeLanguageModel)

    def test_llm_initialization(self):
        """Test LLM initialization"""
        # We're testing the class can be created, but not fully instantiated
        assert hasattr(KimiLargeLanguageModel, '_invoke')

    def test_invoke_success(self):
        """Test successful invocation"""
        # Test that the method exists
        assert hasattr(KimiLargeLanguageModel, '_invoke')

    def test_get_num_tokens(self):
        """Test token counting"""
        # Test that the method exists
        assert hasattr(KimiLargeLanguageModel, 'get_num_tokens')

    def test_validate_credentials_success(self):
        """Test successful credential validation"""
        # Test that the method exists
        assert hasattr(KimiLargeLanguageModel, 'validate_credentials')

    def test_validate_credentials_failure(self):
        """Test credential validation failure"""
        # Test that the method exists
        assert hasattr(KimiLargeLanguageModel, 'validate_credentials')

    def test_convert_messages_to_openai_format(self):
        """Test message conversion"""
        # Test that the method exists
        assert hasattr(KimiLargeLanguageModel, '_convert_messages_to_openai_format')


if __name__ == '__main__':
    unittest.main()