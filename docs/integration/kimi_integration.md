# Kimi AI Integration Guide

## Overview

This guide explains how to integrate Kimi AI from Moonshot AI into Dify. Kimi is a large language model that supports the OpenAI-compatible chat completion API.

## Prerequisites

- A Kimi AI API key from Moonshot AI
- Access to the Kimi API endpoint (default: https://api.moonshot.cn/v1)

## Configuration

### Provider Configuration

To configure the Kimi provider, you need to provide the following credentials:

1. **API Key**: Your Kimi AI API key
2. **Endpoint URL**: The API endpoint URL (default: https://api.moonshot.cn/v1)
3. **Mode**: Set to "chat" for chat completions

### Model Configuration

When configuring a specific Kimi model, you can set:

1. **Context Size**: Maximum context length for the model (default: 32768)
2. **Max Tokens**: Maximum number of tokens to generate (default: 2048)

## Supported Models

Kimi AI currently supports the following models:

- moonshot-v1-8k (8K context window)
- moonshot-v1-32k (32K context window)
- moonshot-v1-128k (128K context window)

## Features

The Kimi integration supports:

- Chat completions
- Streaming responses
- Function calling
- Multi-turn conversations
- System, user, and assistant messages

## Usage

Once configured, you can use Kimi models in your Dify applications just like any other LLM provider. The integration uses the OpenAI-compatible API, so it should work seamlessly with existing Dify workflows.

## Troubleshooting

If you encounter issues with the Kimi integration:

1. Verify your API key is correct and active
2. Check that the endpoint URL is accessible
3. Ensure your model name is correct
4. Verify you have sufficient quota/balance in your Moonshot AI account

For more information, refer to the [Moonshot AI documentation](https://www.moonshot.cn/).