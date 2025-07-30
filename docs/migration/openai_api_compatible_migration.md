# OpenAI API Compatible Provider Migration Guide

## Overview

The `openai_api_compatible` provider has been deprecated and split into two specialized providers:
- `openai_api_chat` - for chat-based completions using `/chat/completions` endpoint
- `openai_api_completion` - for text completions using `/completions` endpoint

This guide will help you migrate from the deprecated `openai_api_compatible` provider to the new specialized providers.

## Why This Change?

The original `openai_api_compatible` provider tried to handle both chat and completion endpoints, which led to:
- Confusion about which endpoint to use
- Inconsistent behavior
- Maintenance complexity
- Performance overhead

The new specialized providers offer:
- Clear separation of concerns
- Better performance
- Easier maintenance
- More predictable behavior

## Migration Steps

### Step 1: Identify Your Use Case

First, determine which type of API you're using:

**Use `openai_api_chat` if you:**
- Use the `/chat/completions` endpoint
- Send conversation-style messages (system, user, assistant)
- Need function calling support
- Work with multi-turn conversations
- Use streaming responses

**Use `openai_api_completion` if you:**
- Use the `/completions` endpoint
- Send single prompts for text completion
- Don't need function calling
- Work with traditional prompt-completion patterns

### Step 2: Update Provider Configuration

#### For Chat Completions

**Before (deprecated):**
```yaml
provider: openai_api_compatible
credentials:
  api_key: "your-api-key"
  endpoint_url: "https://api.openai.com/v1"
  mode: "chat"
```

**After:**
```yaml
provider: openai_api_chat
credentials:
  api_key: "your-api-key"
  endpoint_url: "https://api.openai.com/v1"
  mode: "chat"
```

#### For Text Completions

**Before (deprecated):**
```yaml
provider: openai_api_compatible
credentials:
  api_key: "your-api-key"
  endpoint_url: "https://api.openai.com/v1"
  mode: "completion"
```

**After:**
```yaml
provider: openai_api_completion
credentials:
  api_key: "your-api-key"
  endpoint_url: "https://api.openai.com/v1"
  mode: "completion"
```

### Step 3: Update Code References

#### Python Code Changes

**Before:**
```python
from core.model_runtime.model_providers.openai_api_compatible.llm.llm import OpenAICompatibleLargeLanguageModel

llm = OpenAICompatibleLargeLanguageModel()
```

**After (for chat):**
```python
from core.model_runtime.model_providers.openai_api_chat.llm.llm import OpenAIChatLargeLanguageModel

llm = OpenAIChatLargeLanguageModel()
```

**After (for completion):**
```python
from core.model_runtime.model_providers.openai_api_completion.llm.llm import OpenAICompletionLargeLanguageModel

llm = OpenAICompletionLargeLanguageModel()
```

### Step 4: Update API Calls

The API interface remains the same, but behavior may be more predictable:

```python
# This works the same way with both new providers
result = llm.invoke(
    model="your-model",
    credentials=credentials,
    prompt_messages=messages,
    stream=False
)
```

## Backward Compatibility

The deprecated `openai_api_compatible` provider will continue to work but will:
- Show deprecation warnings
- Automatically route requests to the appropriate new provider
- Be removed in a future version

## Testing Your Migration

### 1. Test Basic Functionality

```python
# Test chat provider
from core.model_runtime.model_providers.openai_api_chat.llm.llm import OpenAIChatLargeLanguageModel

llm = OpenAIChatLargeLanguageModel()
messages = [UserPromptMessage(content="Hello")]
result = llm.invoke("gpt-3.5-turbo", credentials, messages)
print(result.message.content)
```

### 2. Test Error Handling

```python
# Test with invalid credentials
try:
    llm.invoke("invalid-model", invalid_credentials, messages)
except CredentialsValidateFailedError as e:
    print(f"Validation failed: {e}")
```

### 3. Test Streaming

```python
# Test streaming responses
for chunk in llm.invoke("gpt-3.5-turbo", credentials, messages, stream=True):
    print(chunk.delta.message.content, end="")
```

## Common Issues and Solutions

### Issue: "Provider not available" Error

**Cause:** The new provider modules are not properly installed or imported.

**Solution:** Ensure you're using the latest version of Dify and that all dependencies are installed.

### Issue: Mode Validation Errors

**Cause:** Using the wrong mode for the provider.

**Solution:** 
- Use `mode: "chat"` for `openai_api_chat`
- Use `mode: "completion"` for `openai_api_completion`

### Issue: Function Calling Not Working

**Cause:** Function calling is only supported in chat mode.

**Solution:** Use `openai_api_chat` provider for function calling.

### Issue: Performance Degradation

**Cause:** The backward compatibility layer adds overhead.

**Solution:** Migrate to the new specialized providers as soon as possible.

## Timeline

- **Current:** Deprecation warnings are shown
- **Next Release:** Enhanced deprecation warnings
- **Future Release:** Complete removal of `openai_api_compatible`

## Support

If you encounter issues during migration:

1. Check the [Dify documentation](https://docs.dify.ai)
2. Search existing [GitHub issues](https://github.com/langgenius/dify/issues)
3. Create a new issue with details about your problem
4. Join the [Discord community](https://discord.gg/FngNHpbcY7) for help

## Examples

### Complete Migration Example

**Before:**
```python
from core.model_runtime.model_providers.openai_api_compatible.llm.llm import OpenAICompatibleLargeLanguageModel

llm = OpenAICompatibleLargeLanguageModel()

# Chat completion
chat_credentials = {
    "api_key": "sk-...",
    "endpoint_url": "https://api.openai.com/v1",
    "mode": "chat"
}

messages = [
    SystemPromptMessage(content="You are a helpful assistant"),
    UserPromptMessage(content="What is 2+2?")
]

result = llm.invoke("gpt-3.5-turbo", chat_credentials, messages)
```

**After:**
```python
from core.model_runtime.model_providers.openai_api_chat.llm.llm import OpenAIChatLargeLanguageModel

llm = OpenAIChatLargeLanguageModel()

# Chat completion
chat_credentials = {
    "api_key": "sk-...",
    "endpoint_url": "https://api.openai.com/v1",
    "mode": "chat"
}

messages = [
    SystemPromptMessage(content="You are a helpful assistant"),
    UserPromptMessage(content="What is 2+2?")
]

result = llm.invoke("gpt-3.5-turbo", chat_credentials, messages)
```

The interface remains the same, but you get better performance and clearer semantics.