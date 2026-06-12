import assert from "node:assert/strict";
import test from "node:test";

import {
  convertAnthropicToGemini,
  normalizeAnthropicMessagesForGemini,
  type AnthropicRequest,
} from "../src/anthropic-gemini.js";

test("hoists system messages out of messages before Gemini conversion", () => {
  const body: AnthropicRequest = {
    messages: [
      {
        role: "system",
        content: "You are a strict coding assistant.",
      },
      {
        role: "user",
        content: "Say hi.",
      },
    ],
  };

  const normalized = normalizeAnthropicMessagesForGemini(body);

  assert.deepEqual(normalized.system, [
    {
      type: "text",
      text: "You are a strict coding assistant.",
    },
  ]);
  assert.equal(normalized.messages?.length, 1);
  assert.equal(normalized.messages?.[0]?.role, "user");
});

test("builds systemInstruction and excludes system role from Gemini contents", () => {
  const body: AnthropicRequest = {
    system: "Top-level system",
    messages: [
      {
        role: "system",
        content: [
          {
            type: "text",
            text: "Embedded VS Code system",
            cache_control: { type: "ephemeral" },
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Write code",
            cache_control: { type: "ephemeral" },
          },
        ],
      },
    ],
    thinking: { type: "enabled", budget_tokens: 1024 },
  };

  const geminiRequest = convertAnthropicToGemini(body);

  assert.equal(geminiRequest.contents.length, 1);
  assert.equal(geminiRequest.contents[0]?.role, "user");
  assert.deepEqual(geminiRequest.systemInstruction, {
    parts: [{ text: "Top-level system\n\nEmbedded VS Code system" }],
  });
  assert.equal(JSON.stringify(geminiRequest).includes("\"system\""), false);
  assert.equal(JSON.stringify(geminiRequest).includes("cache_control"), false);
  assert.equal("thinking" in geminiRequest, false);
});

test("maps Anthropic tools into Gemini function declarations", () => {
  const body: AnthropicRequest = {
    messages: [
      {
        role: "user",
        content: "Use the weather tool.",
      },
    ],
    tools: [
      {
        name: "get_weather",
        description: "Get the weather in a city",
        input_schema: {
          type: "object",
          properties: {
            city: {
              type: "string",
              description: "City name",
            },
          },
          required: ["city"],
          $schema: "https://json-schema.org/draft/2020-12/schema",
        },
      },
    ],
  };

  const geminiRequest = convertAnthropicToGemini(body);

  assert.deepEqual(geminiRequest.tools, [
    {
      functionDeclarations: [
        {
          name: "get_weather",
          description: "Get the weather in a city",
          parameters: {
            type: "object",
            properties: {
              city: {
                type: "string",
                description: "City name",
              },
            },
            required: ["city"],
          },
        },
      ],
    },
  ]);
});