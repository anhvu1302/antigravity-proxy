import { randomUUID } from "node:crypto";

import {
  sanitizeSchemaForGemini,
  type GeminiRequest,
  type GeminiResponse,
  type JsonObject,
  type JsonPrimitive,
  type JsonValue,
} from "./anthropic-gemini.js";

interface OpenAIToolCall {
  id?: string;
  type?: string;
  function?: {
    name?: string;
    arguments?: string | Record<string, unknown>;
  };
}

interface OpenAIMessage {
  role: string;
  content?: unknown;
  tool_call_id?: string;
  name?: string;
  tool_calls?: OpenAIToolCall[];
}

interface OpenAIToolDefinition {
  type?: string;
  function?: {
    name?: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface OpenAIChatCompletionRequest {
  model?: string;
  messages?: OpenAIMessage[];
  tools?: OpenAIToolDefinition[];
  max_tokens?: number;
  max_completion_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string | string[];
  stream?: boolean;
  [key: string]: unknown;
}

export interface OpenAICompletionRequest {
  model?: string;
  prompt?: string | string[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string | string[];
  stream?: boolean;
  [key: string]: unknown;
}

export interface OpenAIResponsesRequest {
  model?: string;
  input?: unknown;
  instructions?: string;
  tools?: OpenAIToolDefinition[];
  max_output_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string | string[];
  stream?: boolean;
  [key: string]: unknown;
}

interface GeminiBuildOptions {
  systemTexts?: string[];
  tools?: OpenAIToolDefinition[];
  maxOutputTokens?: number;
  temperature?: number;
  topP?: number;
  stop?: string | string[];
}

interface OpenAIUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export function convertOpenAIChatToGemini(body: OpenAIChatCompletionRequest): GeminiRequest {
  const options: GeminiBuildOptions = {};
  const maxOutputTokens = pickFirstNumber(body.max_completion_tokens, body.max_tokens);

  if (body.tools) {
    options.tools = body.tools;
  }

  if (typeof maxOutputTokens === "number") {
    options.maxOutputTokens = maxOutputTokens;
  }

  if (typeof body.temperature === "number") {
    options.temperature = body.temperature;
  }

  if (typeof body.top_p === "number") {
    options.topP = body.top_p;
  }

  if (body.stop !== undefined) {
    options.stop = body.stop;
  }

  return buildGeminiRequestFromMessages(body.messages ?? [], options);
}

export function convertOpenAICompletionToGemini(body: OpenAICompletionRequest): GeminiRequest {
  const options: GeminiBuildOptions = {};

  if (typeof body.max_tokens === "number") {
    options.maxOutputTokens = body.max_tokens;
  }

  if (typeof body.temperature === "number") {
    options.temperature = body.temperature;
  }

  if (typeof body.top_p === "number") {
    options.topP = body.top_p;
  }

  if (body.stop !== undefined) {
    options.stop = body.stop;
  }

  return buildGeminiRequestFromMessages(
    [
      {
        role: "user",
        content: normalizePrompt(body.prompt),
      },
    ],
    options,
  );
}

export function convertOpenAIResponsesToGemini(body: OpenAIResponsesRequest): GeminiRequest {
  const systemTexts = typeof body.instructions === "string" && body.instructions.trim()
    ? [body.instructions.trim()]
    : [];

  const options: GeminiBuildOptions = {
    systemTexts,
  };

  if (body.tools) {
    options.tools = body.tools;
  }

  if (typeof body.max_output_tokens === "number") {
    options.maxOutputTokens = body.max_output_tokens;
  }

  if (typeof body.temperature === "number") {
    options.temperature = body.temperature;
  }

  if (typeof body.top_p === "number") {
    options.topP = body.top_p;
  }

  if (body.stop !== undefined) {
    options.stop = body.stop;
  }

  return buildGeminiRequestFromMessages(normalizeResponsesInputToMessages(body.input), options);
}

export function convertGeminiToOpenAIChatCompletionResponse(
  response: GeminiResponse,
  model: string,
): Record<string, unknown> {
  const candidate = response.candidates?.[0];
  const parts = candidate?.content?.parts ?? [];
  const toolCalls: Array<Record<string, unknown>> = [];
  const textParts: string[] = [];

  for (const part of parts) {
    if (typeof part.text === "string" && part.text.length > 0) {
      textParts.push(part.text);
    }

    if (part.functionCall?.name) {
      toolCalls.push({
        id: `call_${randomUUID().replace(/-/g, "")}`,
        type: "function",
        function: {
          name: part.functionCall.name,
          arguments: JSON.stringify(part.functionCall.args ?? {}),
        },
      });
    }
  }

  const content = textParts.join("\n\n");
  const finishReason = toolCalls.length > 0 ? "tool_calls" : mapGeminiFinishReasonToOpenAI(candidate?.finishReason);

  return {
    id: `chatcmpl_${randomUUID().replace(/-/g, "")}`,
    object: "chat.completion",
    created: getUnixTimestamp(),
    model: response.modelVersion ?? model,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: content || (toolCalls.length > 0 ? null : ""),
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: finishReason,
      },
    ],
    usage: buildOpenAIUsage(response),
  };
}

export function convertGeminiToOpenAICompletionResponse(
  response: GeminiResponse,
  model: string,
): Record<string, unknown> {
  const candidate = response.candidates?.[0];
  const text = collectGeminiText(candidate?.content?.parts ?? []);

  return {
    id: `cmpl_${randomUUID().replace(/-/g, "")}`,
    object: "text_completion",
    created: getUnixTimestamp(),
    model: response.modelVersion ?? model,
    choices: [
      {
        index: 0,
        text,
        logprobs: null,
        finish_reason: mapGeminiFinishReasonToOpenAI(candidate?.finishReason),
      },
    ],
    usage: buildOpenAIUsage(response),
  };
}

export function convertGeminiToOpenAIResponsesResponse(
  response: GeminiResponse,
  model: string,
  metadata?: {
    instructions?: string;
    maxOutputTokens?: number;
  },
): Record<string, unknown> {
  const candidate = response.candidates?.[0];
  const parts = candidate?.content?.parts ?? [];
  const output: Array<Record<string, unknown>> = [];
  const textParts: string[] = [];

  for (const part of parts) {
    if (typeof part.text === "string" && part.text.length > 0) {
      textParts.push(part.text);
    }

    if (part.functionCall?.name) {
      output.push({
        type: "function_call",
        id: `fc_${randomUUID().replace(/-/g, "")}`,
        call_id: `call_${randomUUID().replace(/-/g, "")}`,
        name: part.functionCall.name,
        arguments: JSON.stringify(part.functionCall.args ?? {}),
      });
    }
  }

  const outputText = textParts.join("\n\n");

  if (outputText || output.length === 0) {
    output.push({
      type: "message",
      id: `msg_${randomUUID().replace(/-/g, "")}`,
      status: "completed",
      role: "assistant",
      content: [
        {
          type: "output_text",
          text: outputText,
          annotations: [],
        },
      ],
    });
  }

  return {
    id: `resp_${randomUUID().replace(/-/g, "")}`,
    object: "response",
    created_at: getUnixTimestamp(),
    status: "completed",
    error: null,
    incomplete_details: null,
    instructions: metadata?.instructions ?? null,
    max_output_tokens: metadata?.maxOutputTokens ?? null,
    model: response.modelVersion ?? model,
    output,
    output_text: outputText,
    usage: buildOpenAIUsage(response),
  };
}

function buildGeminiRequestFromMessages(messages: OpenAIMessage[], options: GeminiBuildOptions): GeminiRequest {
  const systemTexts = [...(options.systemTexts ?? [])];
  const contents: GeminiRequest["contents"] = [];
  const toolNameById = new Map<string, string>();

  for (const message of messages) {
    if (message.role === "system" || message.role === "developer") {
      const text = extractTextFromOpenAIContent(message.content).trim();

      if (text) {
        systemTexts.push(text);
      }

      continue;
    }

    const parts = convertOpenAIMessageToGeminiParts(message, toolNameById);

    if (parts.length === 0) {
      continue;
    }

    contents.push({
      role: toGeminiRole(message.role),
      parts,
    });
  }

  const request: GeminiRequest = {
    contents,
  };

  const systemText = systemTexts.filter(Boolean).join("\n\n");

  if (systemText) {
    request.systemInstruction = {
      parts: [{ text: systemText }],
    };
  }

  const generationConfig: GeminiRequest["generationConfig"] = {};

  if (typeof options.maxOutputTokens === "number") {
    generationConfig.maxOutputTokens = options.maxOutputTokens;
  }

  if (typeof options.temperature === "number") {
    generationConfig.temperature = options.temperature;
  }

  if (typeof options.topP === "number") {
    generationConfig.topP = options.topP;
  }

  const stopSequences = normalizeStopSequences(options.stop);

  if (stopSequences.length > 0) {
    generationConfig.stopSequences = stopSequences;
  }

  if (Object.keys(generationConfig).length > 0) {
    request.generationConfig = generationConfig;
  }

  const tools = convertOpenAIToolsToGemini(options.tools);

  if (tools) {
    request.tools = tools;
  }

  return request;
}

function convertOpenAIMessageToGeminiParts(
  message: OpenAIMessage,
  toolNameById: Map<string, string>,
): GeminiRequest["contents"][number]["parts"] {
  const parts: GeminiRequest["contents"][number]["parts"] = [];
  const text = extractTextFromOpenAIContent(message.content);

  if (message.role === "tool") {
    parts.push({
      functionResponse: {
        name: resolveToolResponseName(message, toolNameById),
        response: buildToolResponsePayload(message.content),
      },
    });

    return parts;
  }

  if (text) {
    parts.push({ text });
  }

  if (message.role === "assistant" && Array.isArray(message.tool_calls)) {
    for (const toolCall of message.tool_calls) {
      if (!isFunctionToolCall(toolCall)) {
        continue;
      }

      parts.push({
        functionCall: {
          name: toolCall.function.name,
          args: parseToolCallArguments(toolCall.function.arguments),
        },
      });

      if (toolCall.id) {
        toolNameById.set(toolCall.id, toolCall.function.name);
      }
    }
  }

  return parts;
}

function convertOpenAIToolsToGemini(
  tools: OpenAIToolDefinition[] | undefined,
): GeminiRequest["tools"] | undefined {
  const functionDeclarations = (tools ?? [])
    .filter(isFunctionToolDefinition)
    .map((tool) => ({
      name: tool.function.name,
      ...(tool.function.description ? { description: tool.function.description } : {}),
      parameters: sanitizeSchemaForGemini(tool.function.parameters),
    }));

  if (functionDeclarations.length === 0) {
    return undefined;
  }

  return [{ functionDeclarations }];
}

function normalizeResponsesInputToMessages(input: unknown): OpenAIMessage[] {
  if (typeof input === "string") {
    return [{ role: "user", content: input }];
  }

  if (!Array.isArray(input)) {
    return [];
  }

  const messages: OpenAIMessage[] = [];

  for (const item of input) {
    if (typeof item === "string") {
      messages.push({ role: "user", content: item });
      continue;
    }

    if (!isRecord(item)) {
      continue;
    }

    if (typeof item.role === "string") {
      messages.push({
        role: item.role,
        content: item.content,
        ...(typeof item.tool_call_id === "string" ? { tool_call_id: item.tool_call_id } : {}),
        ...(typeof item.name === "string" ? { name: item.name } : {}),
        ...(Array.isArray(item.tool_calls) ? { tool_calls: item.tool_calls as OpenAIToolCall[] } : {}),
      });
      continue;
    }

    if (item.type === "input_text" && typeof item.text === "string") {
      messages.push({ role: "user", content: item.text });
      continue;
    }

    if (item.type === "output_text" && typeof item.text === "string") {
      messages.push({ role: "assistant", content: item.text });
      continue;
    }

    if (item.type === "function_call_output") {
      messages.push({
        role: "tool",
        content: item.output,
        ...(typeof item.call_id === "string" ? { tool_call_id: item.call_id } : {}),
        ...(typeof item.name === "string" ? { name: item.name } : {}),
      });
    }
  }

  return messages;
}

function normalizePrompt(prompt: unknown): string {
  if (typeof prompt === "string") {
    return prompt;
  }

  if (!Array.isArray(prompt)) {
    return "";
  }

  return prompt
    .map((item) => {
      if (typeof item === "string") {
        return item;
      }

      if (typeof item === "number" || typeof item === "boolean") {
        return String(item);
      }

      return "";
    })
    .filter(Boolean)
    .join("\n\n");
}

function extractTextFromOpenAIContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }

  if (!Array.isArray(content)) {
    return "";
  }

  return content
    .map((part) => {
      if (typeof part === "string") {
        return part;
      }

      if (!isRecord(part)) {
        return "";
      }

      const partType = typeof part.type === "string" ? part.type : "";

      if (["text", "input_text", "output_text"].includes(partType) && typeof part.text === "string") {
        return part.text;
      }

      return "";
    })
    .filter(Boolean)
    .join("\n\n");
}

function buildToolResponsePayload(content: unknown): JsonObject {
  if (isRecord(content)) {
    return sanitizeObject(content);
  }

  const text = typeof content === "string" ? content : extractTextFromOpenAIContent(content);
  const parsed = tryParseJson(text);

  if (isRecord(parsed)) {
    return sanitizeObject(parsed);
  }

  if (!text) {
    return {};
  }

  return {
    content: text,
  };
}

function resolveToolResponseName(message: OpenAIMessage, toolNameById: Map<string, string>): string {
  if (typeof message.name === "string" && message.name.trim()) {
    return message.name.trim();
  }

  if (typeof message.tool_call_id === "string") {
    const mappedName = toolNameById.get(message.tool_call_id);

    if (mappedName) {
      return mappedName;
    }

    if (message.tool_call_id.trim()) {
      return message.tool_call_id.trim();
    }
  }

  return "tool";
}

function parseToolCallArguments(argumentsValue: string | Record<string, unknown> | undefined): JsonObject {
  if (isRecord(argumentsValue)) {
    return sanitizeObject(argumentsValue);
  }

  if (typeof argumentsValue !== "string") {
    return {};
  }

  const parsed = tryParseJson(argumentsValue);
  return isRecord(parsed) ? sanitizeObject(parsed) : {};
}

function normalizeStopSequences(stop: string | string[] | undefined): string[] {
  if (typeof stop === "string") {
    return stop ? [stop] : [];
  }

  if (!Array.isArray(stop)) {
    return [];
  }

  return stop.filter((item): item is string => typeof item === "string" && item.length > 0);
}

function collectGeminiText(parts: Array<{ text?: string }> | undefined): string {
  return (parts ?? [])
    .flatMap((part) => (typeof part.text === "string" && part.text.length > 0 ? [part.text] : []))
    .join("\n\n");
}

function buildOpenAIUsage(response: GeminiResponse): OpenAIUsage {
  const promptTokens = response.usageMetadata?.promptTokenCount ?? 0;
  const completionTokens = response.usageMetadata?.candidatesTokenCount ?? 0;
  const totalTokens = response.usageMetadata?.totalTokenCount ?? (promptTokens + completionTokens);

  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: totalTokens,
  };
}

function mapGeminiFinishReasonToOpenAI(finishReason: string | undefined): string {
  if (finishReason === "MAX_TOKENS") {
    return "length";
  }

  return "stop";
}

function toGeminiRole(role: string): "user" | "model" {
  if (role === "assistant") {
    return "model";
  }

  if (role === "user" || role === "tool") {
    return "user";
  }

  throw new Error(`Unsupported OpenAI message role: ${role}`);
}

function pickFirstNumber(...values: Array<number | undefined>): number | undefined {
  return values.find((value): value is number => typeof value === "number");
}

function getUnixTimestamp(): number {
  return Math.floor(Date.now() / 1000);
}

function isFunctionToolCall(toolCall: OpenAIToolCall): toolCall is OpenAIToolCall & { function: { name: string } } {
  return toolCall.type === "function" && typeof toolCall.function?.name === "string";
}

function isFunctionToolDefinition(
  tool: OpenAIToolDefinition,
): tool is OpenAIToolDefinition & { function: { name: string; description?: string; parameters?: Record<string, unknown> } } {
  return tool.type === "function" && typeof tool.function?.name === "string";
}

function sanitizeObject(value: Record<string, unknown>): JsonObject {
  const cleaned: JsonObject = {};

  for (const [key, entry] of Object.entries(value)) {
    const sanitizedEntry = sanitizeJsonValue(entry);

    if (sanitizedEntry !== undefined) {
      cleaned[key] = sanitizedEntry;
    }
  }

  return cleaned;
}

function sanitizeJsonValue(value: unknown): JsonValue | undefined {
  if (Array.isArray(value)) {
    return value
      .map((entry) => sanitizeJsonValue(entry))
      .filter((entry): entry is JsonValue => entry !== undefined);
  }

  if (isRecord(value)) {
    return sanitizeObject(value);
  }

  if (isJsonPrimitive(value)) {
    return value;
  }

  return undefined;
}

function tryParseJson(value: string): unknown {
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return undefined;
  }
}

function isJsonPrimitive(value: unknown): value is JsonPrimitive {
  return value === null || ["string", "number", "boolean"].includes(typeof value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}