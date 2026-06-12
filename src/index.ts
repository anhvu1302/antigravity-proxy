import "dotenv/config";

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";

import {
  convertAnthropicToGemini,
  convertGeminiToAnthropicResponse,
  normalizeAnthropicMessagesForGemini,
  type AnthropicRequest,
  type GeminiResponse,
} from "./anthropic-gemini.js";
import {
  convertGeminiToOpenAIChatCompletionResponse,
  convertGeminiToOpenAICompletionResponse,
  convertGeminiToOpenAIResponsesResponse,
  convertOpenAIChatToGemini,
  convertOpenAICompletionToGemini,
  convertOpenAIResponsesToGemini,
  type OpenAIChatCompletionRequest,
  type OpenAICompletionRequest,
  type OpenAIResponsesRequest,
} from "./openai-gemini.js";
import {
  isAnthropicMessagesPath,
  isCompatibilityProxyPath,
  isOpenAICompletionsPath,
  isStripSystemParamEnabled,
  stripAnthropicSystemPrompts,
  stripAnthropicSystemPromptsFromJsonString,
  stripSystemFieldFromJson,
  stripSystemFieldFromJsonString,
} from "./proxy-compat.js";
import {
  buildInfoLogEntry,
  buildFailLogEntry,
  buildWarningLogEntry,
  createRequestLogContext,
  logProxyEvent,
} from "./logger.js";

const HOST = process.env.HOST ?? "127.0.0.1";
const PORT = Number(process.env.PORT ?? "8045");
const GEMINI_BASE_URL = process.env.GEMINI_BASE_URL ?? "https://generativelanguage.googleapis.com/v1beta";
const PROXY_FORWARD_BASE_URL = process.env.PROXY_FORWARD_BASE_URL;
const MAX_BODY_SIZE = 10 * 1024 * 1024;
const STRIP_SYSTEM_PARAM = isStripSystemParamEnabled(PORT, process.env.STRIP_SYSTEM_PARAM);

const HOP_BY_HOP_HEADERS = new Set([
  "host",
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "content-length",
  "accept-encoding",
]);

class HttpError extends Error {
  constructor(
    readonly statusCode: number,
    message: string,
  ) {
    super(message);
  }
}

const server = createServer(async (request, response) => {
  try {
    applyCorsHeaders(response);

    if (request.method === "OPTIONS") {
      response.writeHead(204);
      response.end();
      return;
    }

    const url = new URL(request.url ?? "/", `http://${request.headers.host ?? `${HOST}:${PORT}`}`);

    if (request.method === "GET" && (url.pathname === "/health" || url.pathname === "/v1/health")) {
      sendJson(response, 200, { ok: true });
      return;
    }

    if (request.method === "POST" && PROXY_FORWARD_BASE_URL && isCompatibilityProxyPath(url.pathname)) {
      await handleCompatibilityProxyForward(request, response, url);
      return;
    }

    if (request.method === "POST" && isAnthropicMessagesPath(url.pathname)) {
      await handleAnthropicMessages(request, response);
      return;
    }

    if (request.method === "POST" && url.pathname === "/v1/chat/completions") {
      await handleOpenAIChatCompletions(request, response);
      return;
    }

    if (request.method === "POST" && isOpenAICompletionsPath(url.pathname)) {
      await handleOpenAICompletions(request, response);
      return;
    }

    if (request.method === "POST" && url.pathname === "/v1/responses") {
      await handleOpenAIResponses(request, response);
      return;
    }

    if (isGeminiPassthroughPath(url.pathname)) {
      await handleGeminiPassthrough(request, response, url);
      return;
    }

    sendJson(response, 404, {
      error: {
        message: "Route not found.",
      },
    });
  } catch (error) {
    if (error instanceof HttpError) {
      sendJson(response, error.statusCode, {
        error: {
          message: error.message,
        },
      });
      return;
    }

    const message = error instanceof Error ? error.message : "Unexpected server error.";
    sendJson(response, 500, {
      error: {
        message,
      },
    });
  }
});

server.listen(PORT, HOST, () => {
  void logProxyEvent({
    status: "info",
    model: `proxy@${HOST}:${PORT}`,
    message: "listening",
  });
});

async function handleAnthropicMessages(request: IncomingMessage, response: ServerResponse): Promise<void> {
  const logContext = createRequestLogContext("anthropic", request.url ?? "/v1/messages");
  const apiKey = resolveApiKey(request);
  const rawBody = await readRequestBody(request);
  let body: AnthropicRequest;

  try {
    body = JSON.parse(rawBody) as AnthropicRequest;
  } catch {
    await logProxyEvent(buildWarningLogEntry(logContext, undefined, 400, "Request body must be valid JSON."));
    sendAnthropicError(response, 400, "Request body must be valid JSON.");
    return;
  }

  if (STRIP_SYSTEM_PARAM) {
    body = stripAnthropicSystemPrompts(body);
  }

  if (body.stream) {
    await logProxyEvent(buildWarningLogEntry(logContext, resolveModel(body.model), 501, "Streaming responses are not implemented."));
    sendAnthropicError(response, 501, "Streaming responses are not implemented in this proxy yet.");
    return;
  }

  let geminiRequest;

  try {
    body = normalizeAnthropicMessagesForGemini(body);
    geminiRequest = convertAnthropicToGemini(body);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid Anthropic request.";
    await logProxyEvent(buildWarningLogEntry(logContext, resolveModel(body.model), 400, message));
    sendAnthropicError(response, 400, message);
    return;
  }

  const model = resolveModel(body.model);

  if (!model) {
    await logProxyEvent(buildWarningLogEntry(logContext, undefined, 400, "Request body must include a non-empty model."));
    sendAnthropicError(response, 400, "Request body must include a non-empty model.");
    return;
  }

  const upstreamResult = await invokeGeminiGenerateContent(model, apiKey, geminiRequest);

  if (!upstreamResult.upstreamResponse.ok) {
    const message = extractGeminiErrorMessage(upstreamResult.parsedUpstreamBody)
      ?? (upstreamResult.rawUpstreamBody || upstreamResult.upstreamResponse.statusText);
    await logProxyEvent(buildFailLogEntry(logContext, model, upstreamResult.upstreamResponse.status, message));
    sendAnthropicError(response, upstreamResult.upstreamResponse.status, message);
    return;
  }

  await logProxyEvent(buildInfoLogEntry(logContext, model, 200));
  sendJson(
    response,
    200,
    convertGeminiToAnthropicResponse(upstreamResult.parsedUpstreamBody as GeminiResponse, model),
  );
}

async function handleOpenAIChatCompletions(request: IncomingMessage, response: ServerResponse): Promise<void> {
  const logContext = createRequestLogContext("openai-chat", request.url ?? "/v1/chat/completions");
  const apiKey = resolveApiKey(request);
  const rawBody = await readRequestBody(request);
  let body: OpenAIChatCompletionRequest;

  try {
    body = JSON.parse(rawBody) as OpenAIChatCompletionRequest;
  } catch {
    await logProxyEvent(buildWarningLogEntry(logContext, undefined, 400, "Request body must be valid JSON."));
    sendOpenAIError(response, 400, "Request body must be valid JSON.");
    return;
  }

  if (body.stream) {
    await logProxyEvent(buildWarningLogEntry(logContext, resolveModel(body.model), 501, "Streaming responses are not implemented."));
    sendOpenAIError(response, 501, "Streaming responses are not implemented in this proxy yet.");
    return;
  }

  const model = resolveModel(body.model);

  if (!model) {
    await logProxyEvent(buildWarningLogEntry(logContext, undefined, 400, "Request body must include a non-empty model."));
    sendOpenAIError(response, 400, "Request body must include a non-empty model.");
    return;
  }

  let geminiRequest;

  try {
    geminiRequest = convertOpenAIChatToGemini(body);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid OpenAI chat completion request.";
    await logProxyEvent(buildWarningLogEntry(logContext, model, 400, message));
    sendOpenAIError(response, 400, message);
    return;
  }

  const upstreamResult = await invokeGeminiGenerateContent(model, apiKey, geminiRequest);

  if (!upstreamResult.upstreamResponse.ok) {
    const message = extractGeminiErrorMessage(upstreamResult.parsedUpstreamBody)
      ?? (upstreamResult.rawUpstreamBody || upstreamResult.upstreamResponse.statusText);
    await logProxyEvent(buildFailLogEntry(logContext, model, upstreamResult.upstreamResponse.status, message));
    sendOpenAIError(response, upstreamResult.upstreamResponse.status, message);
    return;
  }

  await logProxyEvent(buildInfoLogEntry(logContext, model, 200));
  sendJson(
    response,
    200,
    convertGeminiToOpenAIChatCompletionResponse(upstreamResult.parsedUpstreamBody as GeminiResponse, model),
  );
}

async function handleOpenAICompletions(request: IncomingMessage, response: ServerResponse): Promise<void> {
  const logContext = createRequestLogContext("openai-completion", request.url ?? "/v1/completions");
  const apiKey = resolveApiKey(request);
  const rawBody = await readRequestBody(request);
  let body: OpenAICompletionRequest;

  try {
    body = JSON.parse(rawBody) as OpenAICompletionRequest;
  } catch {
    await logProxyEvent(buildWarningLogEntry(logContext, undefined, 400, "Request body must be valid JSON."));
    sendOpenAIError(response, 400, "Request body must be valid JSON.");
    return;
  }

  if (STRIP_SYSTEM_PARAM) {
    body = stripSystemFieldFromJson(body);
  }

  if (body.stream) {
    await logProxyEvent(buildWarningLogEntry(logContext, resolveModel(body.model), 501, "Streaming responses are not implemented."));
    sendOpenAIError(response, 501, "Streaming responses are not implemented in this proxy yet.");
    return;
  }

  const model = resolveModel(body.model);

  if (!model) {
    await logProxyEvent(buildWarningLogEntry(logContext, undefined, 400, "Request body must include a non-empty model."));
    sendOpenAIError(response, 400, "Request body must include a non-empty model.");
    return;
  }

  let geminiRequest;

  try {
    geminiRequest = convertOpenAICompletionToGemini(body);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid OpenAI completion request.";
    await logProxyEvent(buildWarningLogEntry(logContext, model, 400, message));
    sendOpenAIError(response, 400, message);
    return;
  }

  const upstreamResult = await invokeGeminiGenerateContent(model, apiKey, geminiRequest);

  if (!upstreamResult.upstreamResponse.ok) {
    const message = extractGeminiErrorMessage(upstreamResult.parsedUpstreamBody)
      ?? (upstreamResult.rawUpstreamBody || upstreamResult.upstreamResponse.statusText);
    await logProxyEvent(buildFailLogEntry(logContext, model, upstreamResult.upstreamResponse.status, message));
    sendOpenAIError(response, upstreamResult.upstreamResponse.status, message);
    return;
  }

  await logProxyEvent(buildInfoLogEntry(logContext, model, 200));
  sendJson(
    response,
    200,
    convertGeminiToOpenAICompletionResponse(upstreamResult.parsedUpstreamBody as GeminiResponse, model),
  );
}

async function handleOpenAIResponses(request: IncomingMessage, response: ServerResponse): Promise<void> {
  const logContext = createRequestLogContext("openai-responses", request.url ?? "/v1/responses");
  const apiKey = resolveApiKey(request);
  const rawBody = await readRequestBody(request);
  let body: OpenAIResponsesRequest;

  try {
    body = JSON.parse(rawBody) as OpenAIResponsesRequest;
  } catch {
    await logProxyEvent(buildWarningLogEntry(logContext, undefined, 400, "Request body must be valid JSON."));
    sendOpenAIError(response, 400, "Request body must be valid JSON.");
    return;
  }

  if (body.stream) {
    await logProxyEvent(buildWarningLogEntry(logContext, resolveModel(body.model), 501, "Streaming responses are not implemented."));
    sendOpenAIError(response, 501, "Streaming responses are not implemented in this proxy yet.");
    return;
  }

  const model = resolveModel(body.model);

  if (!model) {
    await logProxyEvent(buildWarningLogEntry(logContext, undefined, 400, "Request body must include a non-empty model."));
    sendOpenAIError(response, 400, "Request body must include a non-empty model.");
    return;
  }

  let geminiRequest;

  try {
    geminiRequest = convertOpenAIResponsesToGemini(body);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid OpenAI responses request.";
    await logProxyEvent(buildWarningLogEntry(logContext, model, 400, message));
    sendOpenAIError(response, 400, message);
    return;
  }

  const upstreamResult = await invokeGeminiGenerateContent(model, apiKey, geminiRequest);

  if (!upstreamResult.upstreamResponse.ok) {
    const message = extractGeminiErrorMessage(upstreamResult.parsedUpstreamBody)
      ?? (upstreamResult.rawUpstreamBody || upstreamResult.upstreamResponse.statusText);
    await logProxyEvent(buildFailLogEntry(logContext, model, upstreamResult.upstreamResponse.status, message));
    sendOpenAIError(response, upstreamResult.upstreamResponse.status, message);
    return;
  }

  await logProxyEvent(buildInfoLogEntry(logContext, model, 200));
  sendJson(
    response,
    200,
    convertGeminiToOpenAIResponsesResponse(
      upstreamResult.parsedUpstreamBody as GeminiResponse,
      model,
      buildResponsesMetadata(body.instructions, body.max_output_tokens),
    ),
  );
}

async function handleGeminiPassthrough(
  request: IncomingMessage,
  response: ServerResponse,
  url: URL,
): Promise<void> {
  const logContext = createRequestLogContext("gemini", url.pathname);
  const apiKey = resolveApiKey(request);
  const rawBody = allowsRequestBody(request.method) ? await readRequestBody(request) : "";
  const upstreamResponse = await fetch(buildGeminiPassthroughUrl(url.pathname, url.search, apiKey), {
    method: request.method ?? "GET",
    headers: buildForwardHeaders(request),
    ...(rawBody ? { body: rawBody } : {}),
  });

  const passthroughModel = extractModelFromGeminiPath(url.pathname);
  const passthroughMessage = upstreamResponse.ok ? undefined : await cloneErrorMessage(upstreamResponse);
  await logProxyEvent(
    upstreamResponse.ok
      ? buildInfoLogEntry(logContext, passthroughModel, upstreamResponse.status)
      : buildFailLogEntry(logContext, passthroughModel, upstreamResponse.status, passthroughMessage ?? upstreamResponse.statusText),
  );
  await relayUpstreamResponse(response, upstreamResponse);
}

async function handleCompatibilityProxyForward(
  request: IncomingMessage,
  response: ServerResponse,
  url: URL,
): Promise<void> {
  const logContext = createRequestLogContext("proxy-forward", url.pathname);
  const rawBody = allowsRequestBody(request.method) ? await readRequestBody(request) : "";
  const model = extractModelFromRequestBody(rawBody);
  const body = !STRIP_SYSTEM_PARAM
    ? rawBody
    : isAnthropicMessagesPath(url.pathname)
      ? stripAnthropicSystemPromptsFromJsonString(rawBody)
      : stripSystemFieldFromJsonString(rawBody);
  const upstreamResponse = await fetch(buildCompatibilityProxyForwardUrl(url.pathname, url.search), {
    method: request.method ?? "GET",
    headers: buildForwardHeaders(request),
    ...(body ? { body } : {}),
  });

  const proxyMessage = upstreamResponse.ok ? undefined : await cloneErrorMessage(upstreamResponse);
  await logProxyEvent(
    upstreamResponse.ok
      ? buildInfoLogEntry(logContext, model, upstreamResponse.status)
      : buildFailLogEntry(logContext, model, upstreamResponse.status, proxyMessage ?? upstreamResponse.statusText),
  );
  await relayUpstreamResponse(response, upstreamResponse);
}

async function invokeGeminiGenerateContent(
  model: string,
  apiKey: string | undefined,
  body: unknown,
): Promise<{
  upstreamResponse: Response;
  rawUpstreamBody: string;
  parsedUpstreamBody: unknown;
}> {
  const upstreamResponse = await fetch(buildGeminiGenerateContentUrl(model, apiKey), {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
  const rawUpstreamBody = await upstreamResponse.text();

  return {
    upstreamResponse,
    rawUpstreamBody,
    parsedUpstreamBody: tryParseJson(rawUpstreamBody),
  };
}

function applyCorsHeaders(response: ServerResponse): void {
  response.setHeader("access-control-allow-origin", "*");
  response.setHeader(
    "access-control-allow-headers",
    "content-type, x-api-key, x-goog-api-key, authorization, anthropic-version, openai-beta",
  );
  response.setHeader("access-control-allow-methods", "GET, POST, OPTIONS");
}

async function readRequestBody(request: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  let totalBytes = 0;

  for await (const chunk of request) {
    const buffer = typeof chunk === "string" ? Buffer.from(chunk) : chunk;
    totalBytes += buffer.length;

    if (totalBytes > MAX_BODY_SIZE) {
      throw new HttpError(413, "Request body is too large.");
    }

    chunks.push(buffer);
  }

  return Buffer.concat(chunks).toString("utf8");
}

function resolveApiKey(request: IncomingMessage): string | undefined {
  const fromHeader = firstNonEmpty([
    getHeaderValue(request.headers["x-goog-api-key"]),
    getHeaderValue(request.headers["x-api-key"]),
    extractBearerToken(getHeaderValue(request.headers.authorization)),
  ]);

  if (fromHeader) {
    return fromHeader;
  }

  return firstNonEmpty([process.env.GEMINI_API_KEY, process.env.GOOGLE_API_KEY]);
}

function resolveModel(requestedModel: unknown): string | undefined {
  if (typeof requestedModel !== "string") {
    return undefined;
  }

  const candidate = stripModelPrefix(requestedModel.trim());

  return candidate || undefined;
}

function buildGeminiGenerateContentUrl(model: string, apiKey: string | undefined): string {
  const baseUrl = resolveConfiguredBaseUrl(GEMINI_BASE_URL);
  const url = new URL(
    joinConfiguredBasePath(baseUrl.pathname, `/models/${encodeURIComponent(model)}:generateContent`),
    baseUrl.origin,
  );

  if (apiKey) {
    url.searchParams.set("key", apiKey);
  }

  return url.toString();
}

function buildGeminiPassthroughUrl(pathname: string, search: string, apiKey: string | undefined): string {
  const baseUrl = resolveConfiguredBaseUrl(GEMINI_BASE_URL);
  const url = new URL(joinConfiguredBasePath(baseUrl.pathname, pathname), baseUrl.origin);
  const searchParams = new URLSearchParams(search);

  for (const [key, value] of searchParams) {
    url.searchParams.append(key, value);
  }

  if (apiKey && !url.searchParams.has("key")) {
    url.searchParams.set("key", apiKey);
  }

  return url.toString();
}

function buildCompatibilityProxyForwardUrl(pathname: string, search: string): string {
  if (!PROXY_FORWARD_BASE_URL) {
    throw new HttpError(500, "Proxy forward base URL is not configured.");
  }

  const baseUrl = resolveConfiguredBaseUrl(PROXY_FORWARD_BASE_URL);
  const url = new URL(joinConfiguredBasePath(baseUrl.pathname, pathname), baseUrl.origin);
  const searchParams = new URLSearchParams(search);

  for (const [key, value] of searchParams) {
    url.searchParams.append(key, value);
  }

  return url.toString();
}

function sendJson(response: ServerResponse, statusCode: number, payload: unknown): void {
  response.writeHead(statusCode, {
    "content-type": "application/json; charset=utf-8",
  });
  response.end(JSON.stringify(payload));
}

function sendAnthropicError(response: ServerResponse, statusCode: number, message: string): void {
  sendJson(response, statusCode, {
    type: "error",
    error: {
      type: statusCode >= 500 ? "api_error" : "invalid_request_error",
      message,
    },
  });
}

function sendOpenAIError(response: ServerResponse, statusCode: number, message: string): void {
  sendJson(response, statusCode, {
    error: {
      message,
      type: statusCode >= 500 ? "server_error" : "invalid_request_error",
      param: null,
      code: null,
    },
  });
}

function extractGeminiErrorMessage(payload: unknown): string | undefined {
  if (!isRecord(payload)) {
    return undefined;
  }

  const error = payload.error;

  if (!isRecord(error)) {
    return undefined;
  }

  return typeof error.message === "string" ? error.message : undefined;
}

function tryParseJson(value: string): unknown {
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return undefined;
  }
}

function getHeaderValue(value: string | string[] | undefined): string | undefined {
  if (typeof value === "string") {
    return value.trim() || undefined;
  }

  if (Array.isArray(value)) {
    return value[0]?.trim() || undefined;
  }

  return undefined;
}

function extractBearerToken(authorization: string | undefined): string | undefined {
  if (!authorization) {
    return undefined;
  }

  const match = authorization.match(/^Bearer\s+(.+)$/i);
  return match?.[1]?.trim() || undefined;
}

function firstNonEmpty(values: Array<string | undefined>): string | undefined {
  return values.find((value): value is string => typeof value === "string" && value.length > 0);
}

function stripModelPrefix(value: string): string {
  return value.replace(/^models\//, "");
}

function isGeminiPassthroughPath(pathname: string): boolean {
  return pathname === "/v1/models"
    || pathname.startsWith("/v1/models/")
    || pathname.startsWith("/v1beta/");
}

function allowsRequestBody(method: string | undefined): boolean {
  return method !== "GET" && method !== "HEAD";
}

function buildForwardHeaders(request: IncomingMessage): Headers {
  const headers = new Headers();

  for (const [name, value] of Object.entries(request.headers)) {
    const lowerName = name.toLowerCase();

    if (!value || HOP_BY_HOP_HEADERS.has(lowerName)) {
      continue;
    }

    if (Array.isArray(value)) {
      headers.set(name, value.join(", "));
      continue;
    }

    headers.set(name, value);
  }

  return headers;
}

async function relayUpstreamResponse(response: ServerResponse, upstreamResponse: Response): Promise<void> {
  const responseHeaders: Record<string, string> = {};

  for (const [name, value] of upstreamResponse.headers) {
    if (!HOP_BY_HOP_HEADERS.has(name.toLowerCase())) {
      responseHeaders[name] = value;
    }
  }

  response.writeHead(upstreamResponse.status, responseHeaders);
  response.end(Buffer.from(await upstreamResponse.arrayBuffer()));
}

function resolveConfiguredBaseUrl(value: string): URL {
  const normalizedValue = /^[a-z][a-z0-9+.-]*:\/\//i.test(value) ? value : `http://${value}`;
  return new URL(normalizedValue);
}

function joinConfiguredBasePath(basePath: string, requestPath: string): string {
  const normalizedBasePath = trimTrailingSlash(basePath);
  const normalizedRequestPath = requestPath.startsWith("/") ? requestPath : `/${requestPath}`;

  if (!normalizedBasePath || normalizedBasePath === "/") {
    return normalizedRequestPath;
  }

  const baseSegments = normalizedBasePath.split("/").filter(Boolean);
  const requestSegments = normalizedRequestPath.split("/").filter(Boolean);
  const lastBaseSegment = baseSegments.at(-1);
  const firstRequestSegment = requestSegments[0];

  if (lastBaseSegment && firstRequestSegment === lastBaseSegment) {
    return `/${[...baseSegments.slice(0, -1), ...requestSegments].join("/")}`;
  }

  return `${normalizedBasePath}${normalizedRequestPath}`;
}

function trimTrailingSlash(value: string): string {
  return value.length > 1 ? value.replace(/\/$/, "") : value;
}

function buildResponsesMetadata(
  instructions: string | undefined,
  maxOutputTokens: number | undefined,
): {
  instructions?: string;
  maxOutputTokens?: number;
} {
  const metadata: {
    instructions?: string;
    maxOutputTokens?: number;
  } = {};

  if (typeof instructions === "string") {
    metadata.instructions = instructions;
  }

  if (typeof maxOutputTokens === "number") {
    metadata.maxOutputTokens = maxOutputTokens;
  }

  return metadata;
}

function extractModelFromRequestBody(rawBody: string): string | undefined {
  if (!rawBody.trim()) {
    return undefined;
  }

  const parsedBody = tryParseJson(rawBody);

  if (!isRecord(parsedBody)) {
    return undefined;
  }

  return resolveModel(parsedBody.model);
}

function extractModelFromGeminiPath(pathname: string): string | undefined {
  const match = pathname.match(/\/models\/([^/:]+)(?::|\/|$)/);
  return match?.[1] ? decodeURIComponent(match[1]) : undefined;
}

async function cloneErrorMessage(response: Response): Promise<string | undefined> {
  const rawBody = await response.clone().text();
  const parsedBody = tryParseJson(rawBody);
  return extractGeminiErrorMessage(parsedBody) ?? (rawBody || undefined);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}