"""
Unit tests for OpenAI API Compatible provider (backward compatibility)
"""

import pytest
import warnings
from unittest.mock import Mock, patch, MagicMock

from core.model_runtime.entities.llm_entities import LLMResult
from core.model_runtime.entities.message_entities import UserPromptMessage
from core.model_runtime.errors.invoke import InvokeError
from core.model_runtime.model_providers.openai_api_compatible.openai_api_compatible import OpenAICompatibleProvider
from core.model_runtime.model_providers.openai_api_compatible.llm.llm import OpenAICompatibleLargeLanguageModel


class TestOpenAICompatibleProvider:
    """Test cases for OpenAICompatibleProvider (backward compatibility)"""

    def setup_method(self):
        """Set up test fixtures"""
        self.provider = OpenAICompatibleProvider()

    def test_provider_initialization(self):
        """Test provider initialization"""
        assert self.provider.provider_name == "openai_api_compatible"
        assert "llm" in self.provider.supported_model_types

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
            "mode": "chat"
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.provider.validate_provider_credentials(credentials)
            
            # Check that deprecation warning was issued
            assert len(w) >= 1
            assert any("deprecated" in str(warning.message).lower() for warning in w)

    def test_validate_model_credentials_with_deprecation_warning(self):
        """Test model credential validation with deprecation warning"""
        credentials = {
            "context_size": "4096",
            "max_tokens": "1000"
        }
        
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
            "mode": "chat"
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
        self.llm = OpenAICompatibleLargeLanguageModel()

    def test_invoke_with_deprecation_warning(self):
        """Test invoke with deprecation warning"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "chat"
        }
        messages = [UserPromptMessage(content="Hello")]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Mock the chat LLM to avoid actual API calls
            with patch.object(self.llm, '_get_chat_llm') as mock_get_chat:
                mock_chat_llm = Mock()
                mock_chat_llm._invoke.return_value = Mock(spec=LLMResult)
                mock_get_chat.return_value = mock_chat_llm
                
                self.llm._invoke("gpt-3.5-turbo", credentials, messages)
                
                # Check that deprecation warning was issued
                assert len(w) >= 1
                assert any("deprecated" in str(warning.message).lower() for warning in w)
                
                # Check that the request was routed to chat provider
                mock_get_chat.assert_called_once()
                mock_chat_llm._invoke.assert_called_once()

    def test_invoke_routes_to_completion_provider(self):
        """Test that invoke routes to completion provider when mode is completion"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "completion"
        }
        messages = [UserPromptMessage(content="Hello")]
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            
            # Mock the completion LLM to avoid actual API calls
            with patch.object(self.llm, '_get_completion_llm') as mock_get_completion:
                mock_completion_llm = Mock()
                mock_completion_llm._invoke.return_value = Mock(spec=LLMResult)
                mock_get_completion.return_value = mock_completion_llm
                
                self.llm._invoke("text-davinci-003", credentials, messages)
                
                # Check that the request was routed to completion provider
                mock_get_completion.assert_called_once()
                mock_completion_llm._invoke.assert_called_once()

    def test_invoke_handles_import_error(self):
        """Test that invoke handles import errors gracefully"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "chat"
        }
        messages = [UserPromptMessage(content="Hello")]
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            
            # Mock import error
            with patch.object(self.llm, '_get_chat_llm') as mock_get_chat:
                mock_get_chat.side_effect = ImportError("Module not found")
                
                with pytest.raises(InvokeError, match="Chat provider not available"):
                    self.llm._invoke("gpt-3.5-turbo", credentials, messages)

    def test_get_num_tokens_routes_correctly(self):
        """Test that get_num_tokens routes to correct provider"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "chat"
        }
        messages = [UserPromptMessage(content="Hello")]
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            
            # Mock the chat LLM
            with patch.object(self.llm, '_get_chat_llm') as mock_get_chat:
                mock_chat_llm = Mock()
                mock_chat_llm.get_num_tokens.return_value = 10
                mock_get_chat.return_value = mock_chat_llm
                
                result = self.llm.get_num_tokens("gpt-3.5-turbo", credentials, messages)
                
                assert result == 10
                mock_get_chat.assert_called_once()
                mock_chat_llm.get_num_tokens.assert_called_once()

    def test_validate_credentials_routes_correctly(self):
        """Test that validate_credentials routes to correct provider"""
        credentials = {
            "api_key": "sk-test1234567890123456789012345678901234567890",
            "endpoint_url": "https://api.openai.com/v1",
            "mode": "chat"
        }
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            
            # Mock the chat LLM
            with patch.object(self.llm, '_get_chat_llm') as mock_get_chat:
                mock_chat_llm = Mock()
                mock_get_chat.return_value = mock_chat_llm
                
                self.llm.validate_credentials("gpt-3.5-turbo", credentials)
                
                mock_get_chat.assert_called_once()
                mock_chat_llm.validate_credentials.assert_called_once_with("gpt-3.5-turbo", credentials)

    def test_lazy_loading_works(self):
        """Test that lazy loading works correctly"""
        # Test that the methods exist and can be called
        assert hasattr(self.llm, '_get_chat_llm')
        assert hasattr(self.llm, '_get_completion_llm')
        
        # Test that they are callable
        assert callable(self.llm._get_chat_llm)
        assert callable(self.llm._get_completion_llm)