import assert from "node:assert/strict";
import test from "node:test";

import {
  isAnthropicMessagesPath,
  isCompatibilityProxyPath,
  isOpenAICompletionsPath,
  isStripSystemParamEnabled,
  stripAnthropicSystemPrompts,
  stripAnthropicSystemPromptsFromJsonString,
  stripSystemFieldFromJson,
  stripSystemFieldFromJsonString,
} from "../src/proxy-compat.js";

test("accepts legacy Anthropic message path aliases", () => {
  assert.equal(isAnthropicMessagesPath("/message"), true);
  assert.equal(isAnthropicMessagesPath("/messages"), true);
  assert.equal(isAnthropicMessagesPath("/v1/message"), true);
  assert.equal(isAnthropicMessagesPath("/v1/messages"), true);
  assert.equal(isAnthropicMessagesPath("/v1/completions"), false);
});

test("accepts legacy OpenAI completion path aliases", () => {
  assert.equal(isOpenAICompletionsPath("/complete"), true);
  assert.equal(isOpenAICompletionsPath("/completions"), true);
  assert.equal(isOpenAICompletionsPath("/v1/complete"), true);
  assert.equal(isOpenAICompletionsPath("/v1/completions"), true);
  assert.equal(isOpenAICompletionsPath("/v1/messages"), false);
});

test("identifies compatibility paths that can be forwarded to the final proxy", () => {
  assert.equal(isCompatibilityProxyPath("/message"), true);
  assert.equal(isCompatibilityProxyPath("/complete"), true);
  assert.equal(isCompatibilityProxyPath("/v1/chat/completions"), false);
});

test("enables system stripping for port 8046 unless explicitly overridden", () => {
  assert.equal(isStripSystemParamEnabled(8046, undefined), true);
  assert.equal(isStripSystemParamEnabled(8045, undefined), false);
  assert.equal(isStripSystemParamEnabled(8046, "false"), false);
  assert.equal(isStripSystemParamEnabled(8045, "true"), true);
});

test("drops the top-level system field without mutating the original body", () => {
  const body = {
    model: "gemini-2.5-flash",
    system: "Do not forward this",
    prompt: "Say hi",
  };

  const sanitized = stripSystemFieldFromJson(body);

  assert.deepEqual(sanitized, {
    model: "gemini-2.5-flash",
    prompt: "Say hi",
  });
  assert.equal(body.system, "Do not forward this");
});

test("drops Anthropic system prompts from top-level and messages without mutating the original body", () => {
  const body = {
    model: "gemini-3-flash-agent",
    system: "Drop this too",
    messages: [
      {
        role: "system",
        content: "Remove this message",
      },
      {
        role: "user",
        content: "Keep this message",
      },
    ],
  };

  const sanitized = stripAnthropicSystemPrompts(body);

  assert.deepEqual(sanitized, {
    model: "gemini-3-flash-agent",
    messages: [
      {
        role: "user",
        content: "Keep this message",
      },
    ],
  });
  assert.equal(body.system, "Drop this too");
  assert.equal(body.messages[0]?.role, "system");
});

test("drops the top-level system field from raw JSON bodies when forwarding", () => {
  const sanitized = stripSystemFieldFromJsonString(JSON.stringify({
    model: "gemini-3-flash-agent",
    system: "Filter me",
    messages: [{ role: "user", content: "ping" }],
  }));

  assert.deepEqual(JSON.parse(sanitized), {
    model: "gemini-3-flash-agent",
    messages: [{ role: "user", content: "ping" }],
  });
});

test("drops Anthropic system prompts from raw JSON bodies when forwarding", () => {
  const sanitized = stripAnthropicSystemPromptsFromJsonString(JSON.stringify({
    model: "gemini-3-flash-agent",
    system: "Filter me",
    messages: [
      { role: "system", content: "Remove me" },
      { role: "user", content: "ping" },
    ],
  }));

  assert.deepEqual(JSON.parse(sanitized), {
    model: "gemini-3-flash-agent",
    messages: [{ role: "user", content: "ping" }],
  });
});