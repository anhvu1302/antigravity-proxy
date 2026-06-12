const ANTHROPIC_MESSAGES_PATHS = new Set([
  "/message",
  "/messages",
  "/v1/message",
  "/v1/messages",
]);

const OPENAI_COMPLETIONS_PATHS = new Set([
  "/complete",
  "/completions",
  "/v1/complete",
  "/v1/completions",
]);

export function isCompatibilityProxyPath(pathname: string): boolean {
  return isAnthropicMessagesPath(pathname) || isOpenAICompletionsPath(pathname);
}

export function isAnthropicMessagesPath(pathname: string): boolean {
  return ANTHROPIC_MESSAGES_PATHS.has(pathname);
}

export function isOpenAICompletionsPath(pathname: string): boolean {
  return OPENAI_COMPLETIONS_PATHS.has(pathname);
}

export function isStripSystemParamEnabled(port: number, envValue: string | undefined): boolean {
  const normalizedEnvValue = envValue?.trim().toLowerCase();

  if (normalizedEnvValue === "1" || normalizedEnvValue === "true" || normalizedEnvValue === "yes") {
    return true;
  }

  if (normalizedEnvValue === "0" || normalizedEnvValue === "false" || normalizedEnvValue === "no") {
    return false;
  }

  return port === 8046;
}

export function stripSystemFieldFromJson<T extends Record<string, unknown>>(body: T): T {
  if (!Object.hasOwn(body, "system")) {
    return body;
  }

  const nextBody = { ...body };
  delete nextBody.system;
  return nextBody as T;
}

export function stripAnthropicSystemPrompts<T extends Record<string, unknown>>(body: T): T {
  const nextBody = stripSystemFieldFromJson(body);

  if (!Array.isArray(nextBody.messages)) {
    return nextBody;
  }

  const filteredMessages = nextBody.messages.filter((message) => !isAnthropicSystemMessage(message));

  if (filteredMessages.length === nextBody.messages.length) {
    return nextBody;
  }

  return {
    ...nextBody,
    messages: filteredMessages,
  } as T;
}

export function stripSystemFieldFromJsonString(rawBody: string): string {
  if (!rawBody.trim()) {
    return rawBody;
  }

  try {
    const parsed = JSON.parse(rawBody) as unknown;

    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      return rawBody;
    }

    return JSON.stringify(stripSystemFieldFromJson(parsed as Record<string, unknown>));
  } catch {
    return rawBody;
  }
}

export function stripAnthropicSystemPromptsFromJsonString(rawBody: string): string {
  if (!rawBody.trim()) {
    return rawBody;
  }

  try {
    const parsed = JSON.parse(rawBody) as unknown;

    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      return rawBody;
    }

    return JSON.stringify(stripAnthropicSystemPrompts(parsed as Record<string, unknown>));
  } catch {
    return rawBody;
  }
}

function isAnthropicSystemMessage(value: unknown): value is Record<string, unknown> {
  return typeof value === "object"
    && value !== null
    && !Array.isArray(value)
    && (value as Record<string, unknown>).role === "system";
}