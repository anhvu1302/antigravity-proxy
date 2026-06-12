import assert from "node:assert/strict";
import test from "node:test";

import {
  convertGeminiToOpenAIChatCompletionResponse,
  convertGeminiToOpenAIResponsesResponse,
  convertOpenAIChatToGemini,
  convertOpenAICompletionToGemini,
  convertOpenAIResponsesToGemini,
  type OpenAIChatCompletionRequest,
  type OpenAIResponsesRequest,
} from "../src/openai-gemini.js";

test("converts OpenAI chat messages and tools into Gemini generateContent shape", () => {
  const body: OpenAIChatCompletionRequest = {
    model: "gemini-2.5-flash",
    messages: [
      {
        role: "system",
        content: "System instructions",
      },
      {
        role: "developer",
        content: "Developer instructions",
      },
      {
        role: "user",
        content: [{ type: "text", text: "Find weather" }],
      },
      {
        role: "assistant",
        tool_calls: [
          {
            id: "call_weather",
            type: "function",
            function: {
              name: "get_weather",
              arguments: "{\"city\":\"Hanoi\"}",
            },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: "call_weather",
        content: "{\"temperature\":30}",
      },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Get weather by city",
          parameters: {
            type: "object",
            properties: {
              city: {
                type: "string",
              },
            },
            required: ["city"],
          },
        },
      },
    ],
  };

  const geminiRequest = convertOpenAIChatToGemini(body);

  assert.equal(geminiRequest.contents.length, 3);
  assert.deepEqual(geminiRequest.systemInstruction, {
    parts: [{ text: "System instructions\n\nDeveloper instructions" }],
  });
  assert.deepEqual(geminiRequest.contents[0], {
    role: "user",
    parts: [{ text: "Find weather" }],
  });
  assert.deepEqual(geminiRequest.contents[1], {
    role: "model",
    parts: [
      {
        functionCall: {
          name: "get_weather",
          args: { city: "Hanoi" },
        },
      },
    ],
  });
  assert.deepEqual(geminiRequest.contents[2], {
    role: "user",
    parts: [
      {
        functionResponse: {
          name: "get_weather",
          response: { temperature: 30 },
        },
      },
    ],
  });
});

test("converts Gemini function calls into OpenAI chat completion tool_calls", () => {
  const response = convertGeminiToOpenAIChatCompletionResponse(
    {
      candidates: [
        {
          content: {
            parts: [
              {
                functionCall: {
                  name: "get_weather",
                  args: { city: "Hanoi" },
                },
              },
            ],
          },
          finishReason: "STOP",
        },
      ],
      usageMetadata: {
        promptTokenCount: 10,
        candidatesTokenCount: 4,
        totalTokenCount: 14,
      },
    },
    "gemini-2.5-flash",
  );

  const choices = response.choices as Array<Record<string, unknown>>;
  const firstChoice = choices[0] as Record<string, unknown>;
  const message = firstChoice.message as Record<string, unknown>;
  const toolCalls = message.tool_calls as Array<Record<string, unknown>>;
  const firstToolCall = toolCalls[0] as Record<string, unknown>;
  const toolCallFunction = firstToolCall.function as Record<string, unknown>;

  assert.equal(firstChoice.finish_reason, "tool_calls");
  assert.equal(message.role, "assistant");
  assert.equal(message.content, null);
  assert.equal(toolCallFunction.name, "get_weather");
  assert.equal(toolCallFunction.arguments, '{"city":"Hanoi"}');
  assert.deepEqual(response.usage, {
    prompt_tokens: 10,
    completion_tokens: 4,
    total_tokens: 14,
  });
});

test("converts OpenAI legacy completions prompt arrays into a single Gemini user message", () => {
  const geminiRequest = convertOpenAICompletionToGemini({
    model: "gemini-2.5-flash",
    prompt: ["Line one", "Line two"],
    max_tokens: 128,
  });

  assert.deepEqual(geminiRequest.contents, [
    {
      role: "user",
      parts: [{ text: "Line one\n\nLine two" }],
    },
  ]);
  assert.deepEqual(geminiRequest.generationConfig, {
    maxOutputTokens: 128,
  });
});

test("converts OpenAI responses input and Gemini output into responses API shape", () => {
  const body: OpenAIResponsesRequest = {
    model: "gemini-2.5-flash",
    instructions: "Answer briefly",
    input: [
      {
        role: "user",
        content: [{ type: "input_text", text: "Say hi" }],
      },
    ],
    max_output_tokens: 64,
  };

  const geminiRequest = convertOpenAIResponsesToGemini(body);

  assert.deepEqual(geminiRequest.systemInstruction, {
    parts: [{ text: "Answer briefly" }],
  });
  assert.deepEqual(geminiRequest.contents, [
    {
      role: "user",
      parts: [{ text: "Say hi" }],
    },
  ]);

  const response = convertGeminiToOpenAIResponsesResponse(
    {
      candidates: [
        {
          content: {
            parts: [{ text: "Hi" }],
          },
          finishReason: "STOP",
        },
      ],
      usageMetadata: {
        promptTokenCount: 8,
        candidatesTokenCount: 2,
        totalTokenCount: 10,
      },
    },
    "gemini-2.5-flash",
    body.instructions || typeof body.max_output_tokens === "number"
      ? {
          ...(body.instructions ? { instructions: body.instructions } : {}),
          ...(typeof body.max_output_tokens === "number"
            ? { maxOutputTokens: body.max_output_tokens }
            : {}),
        }
      : undefined,
  );

  assert.equal(response.object, "response");
  assert.equal(response.instructions, "Answer briefly");
  assert.equal(response.max_output_tokens, 64);
  assert.equal(response.output_text, "Hi");
  assert.deepEqual(response.usage, {
    prompt_tokens: 8,
    completion_tokens: 2,
    total_tokens: 10,
  });
});