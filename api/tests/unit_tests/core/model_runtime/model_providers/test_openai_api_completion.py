"""
Unit tests for OpenAI API Completion provider
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import httpx

from core.model_runtime.entities.llm_entities import LLMResult, LLMResultChunk
from core.model_runtime.entities.message_entities import (
    UserPromptMessage,
    SystemPromptMessage,
    AssistantPromptMessage,
)
from core.model_runtime.errors.validate import CredentialsValidateFailedError
from core.model_runtime.errors.invoke import (
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from core.model_runtime.model_providers.openai_api_completion.openai_api_completion import OpenAICompletionProvider
from core.model_runtime.model_providers.openai_api_completion.llm.llm import OpenAICompletionLargeLanguageModel


class TestOpenAICompletionProvider:
    """Test cases for OpenAICompletionProvider"""

    def setup_method(self):
        """Set up test fixtures"""
        self.provider = OpenAICompletionProvider()

    def test_provider_initialization(self):
        """Test provider initialization"""
        assert self.provider.provider_name == "openai_api_completion"
        assert "llm" in self.provider.supported_model_types

    def test_get_provider_schema(self):
        """Test provider schema"""
        schema = self.provider.get_provider_schema()
        assert schema["provider"] == "openai_api_completion"
        assert "credential_form_schemas" in schema["provider_credential_schema"]

    def test_validate_provider_credentials_success(self):
        """Test successful credential validation"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "completion"
        }
        # Should not raise any exception
        self.provider.validate_provider_credentials(credentials)

    def test_validate_provider_credentials_missing_api_key(self):
        """Test credential validation with missing API key"""
        credentials = {
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "completion"
        }
        with pytest.raises(CredentialsValidateFailedError, match="Missing required field: api_key"):
            self.provider.validate_provider_credentials(credentials)

    def test_validate_provider_credentials_invalid_api_key(self):
        """Test credential validation with invalid API key"""
        credentials = {
            "api_key": "short",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "completion"
        }
        with pytest.raises(CredentialsValidateFailedError, match="at least 10 characters"):
            self.provider.validate_provider_credentials(credentials)

    def test_validate_provider_credentials_invalid_endpoint(self):
        """Test credential validation with invalid endpoint"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "invalid-url",
            "mode": "completion"
        }
        with pytest.raises(CredentialsValidateFailedError, match="valid HTTP/HTTPS URL"):
            self.provider.validate_provider_credentials(credentials)

    def test_validate_provider_credentials_wrong_mode(self):
        """Test credential validation with wrong mode"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "chat"
        }
        with pytest.raises(CredentialsValidateFailedError, match="only supports 'completion' mode"):
            self.provider.validate_provider_credentials(credentials)

    def test_validate_model_credentials_success(self):
        """Test successful model credential validation"""
        credentials = {
            "context_size": "4096",
            "max_tokens": "1000"
        }
        # Should not raise any exception
        self.provider.validate_model_credentials("text-davinci-003", credentials)

    def test_validate_model_credentials_invalid_context_size(self):
        """Test model credential validation with invalid context size"""
        credentials = {
            "context_size": "invalid",
            "max_tokens": "1000"
        }
        with pytest.raises(CredentialsValidateFailedError, match="valid integer"):
            self.provider.validate_model_credentials("text-davinci-003", credentials)

    def test_validate_model_credentials_invalid_max_tokens(self):
        """Test model credential validation with invalid max tokens"""
        credentials = {
            "context_size": "4096",
            "max_tokens": "-1"
        }
        with pytest.raises(CredentialsValidateFailedError, match="between 1 and 100,000"):
            self.provider.validate_model_credentials("text-davinci-003", credentials)


class TestOpenAICompletionLargeLanguageModel:
    """Test cases for OpenAICompletionLargeLanguageModel"""

    def setup_method(self):
        """Set up test fixtures"""
        self.llm = OpenAICompletionLargeLanguageModel()

    def test_invoke_wrong_mode(self):
        """Test invoke with wrong mode"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "chat"
        }
        messages = [UserPromptMessage(content="Hello")]
        
        with pytest.raises(CredentialsValidateFailedError, match="only supports completion mode"):
            self.llm._invoke("text-davinci-003", credentials, messages)

    def test_invoke_with_tools_warning(self):
        """Test invoke with tools (should log warning)"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "completion"
        }
        messages = [UserPromptMessage(content="Hello")]
        tools = [{"name": "test_tool"}]
        
        with patch.object(self.llm, '_text_completion_request') as mock_request:
            mock_request.return_value = Mock()
            self.llm._invoke("text-davinci-003", credentials, messages, tools=tools)
            mock_request.assert_called_once()

    @patch('httpx.Client')
    def test_invoke_success(self, mock_client):
        """Test successful invoke"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "completion"
        }
        messages = [UserPromptMessage(content="Hello")]
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "text": "Hello there!",
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8
            }
        }
        
        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_client_instance
        
        result = self.llm._invoke("text-davinci-003", credentials, messages, stream=False)
        
        assert isinstance(result, LLMResult)
        assert result.message.content == "Hello there!"

    @patch('httpx.Client')
    def test_invoke_authorization_error(self, mock_client):
        """Test invoke with authorization error"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "completion"
        }
        messages = [UserPromptMessage(content="Hello")]
        
        # Mock authorization error
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key"}
        }
        
        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_client_instance
        
        with pytest.raises(InvokeAuthorizationError, match="Invalid API key"):
            self.llm._invoke("text-davinci-003", credentials, messages, stream=False)

    def test_convert_messages_to_prompt(self):
        """Test message to prompt conversion"""
        messages = [
            SystemPromptMessage(content="You are a helpful assistant"),
            UserPromptMessage(content="Hello")
        ]
        
        result = self.llm._convert_messages_to_prompt(messages)
        
        # Should combine messages into a single prompt string
        assert isinstance(result, str)
        assert "You are a helpful assistant" in result
        assert "Hello" in result

    def test_get_num_tokens(self):
        """Test token counting"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "completion"
        }
        messages = [UserPromptMessage(content="Hello world")]
        
        # This is a basic test - actual token counting would depend on the model
        result = self.llm.get_num_tokens("text-davinci-003", credentials, messages)
        assert isinstance(result, int)
        assert result >= 0