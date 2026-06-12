import { randomUUID } from "node:crypto";

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export type JsonObject = { [key: string]: JsonValue };

export interface AnthropicTextBlock {
  type: "text";
  text: string;
  cache_control?: unknown;
}

export interface AnthropicToolUseBlock {
  type: "tool_use";
  id?: string;
  name: string;
  input?: Record<string, unknown>;
  cache_control?: unknown;
}

export interface AnthropicToolResultBlock {
  type: "tool_result";
  tool_use_id?: string;
  name?: string;
  content: AnthropicContent;
  is_error?: boolean;
  cache_control?: unknown;
}

export type AnthropicContentBlock =
  | AnthropicTextBlock
  | AnthropicToolUseBlock
  | AnthropicToolResultBlock
  | { type: string; [key: string]: unknown };

export type AnthropicContent = string | Array<string | AnthropicContentBlock>;

export interface AnthropicMessage {
  role: string;
  content: AnthropicContent;
}

export interface AnthropicTool {
  name: string;
  description?: string;
  input_schema?: Record<string, unknown>;
}

export interface AnthropicRequest {
  model?: string;
  system?: string | AnthropicTextBlock[];
  messages?: AnthropicMessage[];
  tools?: AnthropicTool[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop_sequences?: string[];
  stream?: boolean;
  [key: string]: unknown;
}

export interface GeminiTextPart {
  text: string;
}

export interface GeminiFunctionCallPart {
  functionCall: {
    name: string;
    args: JsonObject;
  };
}

export interface GeminiFunctionResponsePart {
  functionResponse: {
    name: string;
    response: JsonObject;
  };
}

export type GeminiPart = GeminiTextPart | GeminiFunctionCallPart | GeminiFunctionResponsePart;

export interface GeminiRequest {
  contents: Array<{
    role: "user" | "model";
    parts: GeminiPart[];
  }>;
  systemInstruction?: {
    parts: GeminiTextPart[];
  };
  generationConfig?: {
    maxOutputTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };
  tools?: Array<{
    functionDeclarations: Array<{
      name: string;
      description?: string;
      parameters: JsonObject;
    }>;
  }>;
}

export interface GeminiResponse {
  candidates?: Array<{
    content?: {
      parts?: Array<{
        text?: string;
        functionCall?: {
          name?: string;
          args?: JsonObject;
        };
      }>;
    };
    finishReason?: string;
  }>;
  usageMetadata?: {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    totalTokenCount?: number;
  };
  modelVersion?: string;
}

export interface AnthropicResponse {
  id: string;
  type: "message";
  role: "assistant";
  model: string;
  content: Array<
    | {
        type: "text";
        text: string;
      }
    | {
        type: "tool_use";
        id: string;
        name: string;
        input: JsonObject;
      }
  >;
  stop_reason: "end_turn" | "max_tokens" | "tool_use" | null;
  stop_sequence: null;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

type AnthropicAssistantContentBlock = AnthropicResponse["content"][number];

const FUNCTION_SCHEMA_KEYS = new Set([
  "type",
  "format",
  "description",
  "nullable",
  "enum",
  "items",
  "properties",
  "required",
  "additionalProperties",
  "minimum",
  "maximum",
  "minItems",
  "maxItems",
  "minLength",
  "maxLength",
  "pattern",
  "anyOf",
  "oneOf",
  "allOf",
]);

export function extractTextContent(content: AnthropicContent | undefined): string {
  if (!content) {
    return "";
  }

  if (typeof content === "string") {
    return content;
  }

  return content
    .map((part) => {
      if (typeof part === "string") {
        return part;
      }

      if (isAnthropicTextBlock(part)) {
        return part.text;
      }

      if (isAnthropicToolResultBlock(part)) {
        return extractTextContent(part.content);
      }

      return "";
    })
    .filter(Boolean)
    .join("\n\n");
}

export function normalizeAnthropicMessagesForGemini(body: AnthropicRequest): AnthropicRequest {
  const nextBody = structuredClone(body) as AnthropicRequest;
  const systemBlocks: AnthropicTextBlock[] = [];

  collectSystemBlocks(systemBlocks, nextBody.system);

  const cleanMessages: AnthropicMessage[] = [];

  for (const message of nextBody.messages ?? []) {
    if (message.role === "system") {
      const text = extractTextContent(message.content).trim();

      if (text) {
        systemBlocks.push({
          type: "text",
          text,
        });
      }

      continue;
    }

    cleanMessages.push(message);
  }

  nextBody.system = systemBlocks;
  nextBody.messages = cleanMessages;

  return nextBody;
}

export function toGeminiRole(role: string): "user" | "model" {
  if (role === "user") {
    return "user";
  }

  if (role === "assistant") {
    return "model";
  }

  throw new Error(`Unsupported Gemini content role: ${role}`);
}

export function convertContentToGeminiParts(content: AnthropicContent): GeminiPart[] {
  if (typeof content === "string") {
    return [{ text: content }];
  }

  const parts: GeminiPart[] = [];

  for (const block of content) {
    if (typeof block === "string") {
      parts.push({ text: block });
      continue;
    }

    if (isAnthropicTextBlock(block)) {
      parts.push({ text: block.text });
      continue;
    }

    if (isAnthropicToolUseBlock(block)) {
      parts.push({
        functionCall: {
          name: block.name,
          args: sanitizeObject(block.input ?? {}),
        },
      });
      continue;
    }

    if (isAnthropicToolResultBlock(block)) {
      parts.push({
        functionResponse: {
          name: typeof block.name === "string"
            ? block.name
            : typeof block.tool_use_id === "string"
              ? block.tool_use_id
              : "tool_result",
          response: buildToolResultResponse(block),
        },
      });
      continue;
    }

    throw new Error(`Unsupported Anthropic content block type: ${block.type}`);
  }

  return parts;
}

export function convertAnthropicToGemini(body: AnthropicRequest): GeminiRequest {
  const normalizedBody = normalizeAnthropicMessagesForGemini(body);

  const contents = (normalizedBody.messages ?? []).map((message) => ({
    role: toGeminiRole(message.role),
    parts: convertContentToGeminiParts(message.content),
  }));

  const systemBlocks = Array.isArray(normalizedBody.system) ? normalizedBody.system : [];

  const systemText = systemBlocks
    .map((block) => block?.text ?? "")
    .filter(Boolean)
    .join("\n\n");

  const geminiRequest: GeminiRequest = {
    contents,
    ...(systemText
      ? {
          systemInstruction: {
            parts: [{ text: systemText }],
          },
        }
      : {}),
  };

  const generationConfig: GeminiRequest["generationConfig"] = {};

  if (typeof normalizedBody.max_tokens === "number") {
    generationConfig.maxOutputTokens = normalizedBody.max_tokens;
  }

  if (typeof normalizedBody.temperature === "number") {
    generationConfig.temperature = normalizedBody.temperature;
  }

  if (typeof normalizedBody.top_p === "number") {
    generationConfig.topP = normalizedBody.top_p;
  }

  const stopSequences = normalizedBody.stop_sequences?.filter(
    (sequence): sequence is string => typeof sequence === "string" && sequence.length > 0,
  );

  if (stopSequences && stopSequences.length > 0) {
    generationConfig.stopSequences = stopSequences;
  }

  if (Object.keys(generationConfig).length > 0) {
    geminiRequest.generationConfig = generationConfig;
  }

  const geminiTools = convertToolsToGeminiTools(normalizedBody.tools);

  if (geminiTools) {
    geminiRequest.tools = geminiTools;
  }

  return geminiRequest;
}

export function convertGeminiToAnthropicResponse(response: GeminiResponse, model: string): AnthropicResponse {
  const candidate = response.candidates?.[0];
  const parts = candidate?.content?.parts ?? [];
  const content: AnthropicAssistantContentBlock[] = [];

  for (const part of parts) {
    if (typeof part.text === "string" && part.text.length > 0) {
      content.push({
        type: "text",
        text: part.text,
      });
      continue;
    }

    if (part.functionCall?.name) {
      content.push({
        type: "tool_use",
        id: `toolu_${randomUUID().replace(/-/g, "")}`,
        name: part.functionCall.name,
        input: sanitizeObject(part.functionCall.args ?? {}),
      });
    }
  }

  const stopReason = content.some((block) => block.type === "tool_use")
    ? "tool_use"
    : mapFinishReason(candidate?.finishReason);

  return {
    id: `msg_${randomUUID().replace(/-/g, "")}`,
    type: "message",
    role: "assistant",
    model: response.modelVersion ?? model,
    content: content.length > 0 ? content : [{ type: "text", text: "" }],
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: response.usageMetadata?.promptTokenCount ?? 0,
      output_tokens: response.usageMetadata?.candidatesTokenCount ?? 0,
    },
  };
}

function collectSystemBlocks(target: AnthropicTextBlock[], system: AnthropicRequest["system"]): void {
  if (typeof system === "string") {
    const text = system.trim();

    if (text) {
      target.push({
        type: "text",
        text,
      });
    }

    return;
  }

  if (!Array.isArray(system)) {
    return;
  }

  for (const block of system) {
    const text = typeof block?.text === "string" ? block.text.trim() : "";

    if (text) {
      target.push({
        type: "text",
        text,
      });
    }
  }
}

function convertToolsToGeminiTools(tools: AnthropicTool[] | undefined): GeminiRequest["tools"] | undefined {
  if (!tools || tools.length === 0) {
    return undefined;
  }

  return [
    {
      functionDeclarations: tools.map((tool) => ({
        name: tool.name,
        ...(tool.description ? { description: tool.description } : {}),
        parameters: sanitizeSchemaForGemini(tool.input_schema),
      })),
    },
  ];
}

function buildToolResultResponse(block: AnthropicToolResultBlock): JsonObject {
  const text = extractTextContent(block.content);
  const response: JsonObject = {};

  if (text) {
    response.content = text;
  }

  if (block.is_error) {
    response.isError = true;
  }

  return response;
}

export function sanitizeSchemaForGemini(schema: Record<string, unknown> | undefined): JsonObject {
  if (!isRecord(schema)) {
    return {
      type: "object",
      properties: {},
    };
  }

  const cleaned = sanitizeSchemaNode(schema);

  if (!isRecord(cleaned)) {
    return {
      type: "object",
      properties: {},
    };
  }

  if (cleaned.type !== "object" && typeof cleaned.type !== "string") {
    cleaned.type = "object";
  }

  if (cleaned.type === "object" && !isRecord(cleaned.properties)) {
    cleaned.properties = {};
  }

  return cleaned;
}

function sanitizeSchemaNode(value: unknown): JsonValue | undefined {
  if (Array.isArray(value)) {
    const items = value
      .map((entry) => sanitizeSchemaNode(entry))
      .filter((entry): entry is JsonValue => entry !== undefined);

    return items;
  }

  if (!isRecord(value)) {
    return isJsonPrimitive(value) ? value : undefined;
  }

  const cleaned: JsonObject = {};

  for (const [key, entry] of Object.entries(value)) {
    if (!FUNCTION_SCHEMA_KEYS.has(key)) {
      continue;
    }

    if (key === "properties") {
      const sanitizedProperties = sanitizeSchemaProperties(entry);

      if (sanitizedProperties) {
        cleaned[key] = sanitizedProperties;
      }

      continue;
    }

    const sanitizedEntry = sanitizeSchemaNode(entry);

    if (sanitizedEntry !== undefined) {
      cleaned[key] = sanitizedEntry;
    }
  }

  return cleaned;
}

function sanitizeSchemaProperties(value: unknown): JsonObject | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const cleaned: JsonObject = {};

  for (const [key, entry] of Object.entries(value)) {
    const sanitizedEntry = sanitizeSchemaNode(entry);

    if (sanitizedEntry !== undefined) {
      cleaned[key] = sanitizedEntry;
    }
  }

  return cleaned;
}

function sanitizeObject(value: unknown): JsonObject {
  if (!isRecord(value)) {
    return {};
  }

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

function mapFinishReason(finishReason: string | undefined): AnthropicResponse["stop_reason"] {
  if (finishReason === "MAX_TOKENS") {
    return "max_tokens";
  }

  return "end_turn";
}

function isAnthropicTextBlock(block: AnthropicContentBlock | string): block is AnthropicTextBlock {
  return typeof block !== "string" && block.type === "text" && typeof block.text === "string";
}

function isAnthropicToolUseBlock(block: AnthropicContentBlock | string): block is AnthropicToolUseBlock {
  return typeof block !== "string" && block.type === "tool_use" && typeof block.name === "string";
}

function isAnthropicToolResultBlock(block: AnthropicContentBlock | string): block is AnthropicToolResultBlock {
  return typeof block !== "string" && block.type === "tool_result" && "content" in block;
}

function isJsonPrimitive(value: unknown): value is JsonPrimitive {
  return value === null || ["string", "number", "boolean"].includes(typeof value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}