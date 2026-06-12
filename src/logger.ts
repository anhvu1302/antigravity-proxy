import { appendFile, mkdir, readdir, rm } from "node:fs/promises";
import path from "node:path";

const LOG_DIR = path.resolve(process.cwd(), "logs");
const LOG_PREFIX = "proxy-";
const LOG_SUFFIX = ".log";
const LOG_RETENTION_DAYS = 7;

const ANSI = {
  reset: "\u001B[0m",
  dim: "\u001B[2m",
  cyan: "\u001B[36m",
  green: "\u001B[32m",
  red: "\u001B[31m",
  yellow: "\u001B[33m",
} as const;

export type ProxyLogStatus = "info" | "warning" | "fail";

export interface ProxyLogEntry {
  status: ProxyLogStatus;
  model?: string;
  protocol?: string;
  route?: string;
  statusCode?: number;
  durationMs?: number;
  message?: string;
}

let retentionPromise: Promise<void> | undefined;

export async function logProxyEvent(entry: ProxyLogEntry): Promise<void> {
  const timestamp = new Date();
  await ensureLogRetention();

  const terminalLine = formatTerminalLine(timestamp, entry);
  const fileLine = formatFileLine(timestamp, entry);

  console.log(terminalLine);
  await appendLogLine(timestamp, fileLine);
}

export function createRequestLogContext(protocol: string, route: string, startedAt = Date.now()): {
  protocol: string;
  route: string;
  startedAt: number;
} {
  return {
    protocol,
    route,
    startedAt,
  };
}

export function buildInfoLogEntry(
  context: { protocol: string; route: string; startedAt: number },
  model: string | undefined,
  statusCode: number,
): ProxyLogEntry {
  return {
    status: "info",
    protocol: context.protocol,
    route: context.route,
    statusCode,
    durationMs: Date.now() - context.startedAt,
    ...(model ? { model } : {}),
  };
}

export function buildWarningLogEntry(
  context: { protocol: string; route: string; startedAt: number },
  model: string | undefined,
  statusCode: number,
  message: string,
): ProxyLogEntry {
  return {
    status: "warning",
    protocol: context.protocol,
    route: context.route,
    statusCode,
    durationMs: Date.now() - context.startedAt,
    message,
    ...(model ? { model } : {}),
  };
}

export function buildFailLogEntry(
  context: { protocol: string; route: string; startedAt: number },
  model: string | undefined,
  statusCode: number,
  message: string,
): ProxyLogEntry {
  return {
    status: "fail",
    protocol: context.protocol,
    route: context.route,
    statusCode,
    durationMs: Date.now() - context.startedAt,
    message,
    ...(model ? { model } : {}),
  };
}

export function getLogFilePath(date: Date): string {
  return path.join(LOG_DIR, `${LOG_PREFIX}${formatDate(date)}${LOG_SUFFIX}`);
}

export function shouldDeleteLogFile(fileName: string, now: Date): boolean {
  if (!fileName.startsWith(LOG_PREFIX) || !fileName.endsWith(LOG_SUFFIX)) {
    return false;
  }

  const rawDate = fileName.slice(LOG_PREFIX.length, -LOG_SUFFIX.length);
  const fileDate = parseLogDate(rawDate);

  if (!fileDate) {
    return false;
  }

  return diffInUtcDays(fileDate, now) >= LOG_RETENTION_DAYS;
}

async function appendLogLine(date: Date, line: string): Promise<void> {
  await mkdir(LOG_DIR, { recursive: true });
  await appendFile(getLogFilePath(date), `${line}\n`, "utf8");
}

async function ensureLogRetention(): Promise<void> {
  retentionPromise ??= cleanupOldLogs().finally(() => {
    retentionPromise = undefined;
  });

  await retentionPromise;
}

async function cleanupOldLogs(now = new Date()): Promise<void> {
  await mkdir(LOG_DIR, { recursive: true });
  const entries = await readdir(LOG_DIR, { withFileTypes: true });

  await Promise.all(entries
    .filter((entry) => entry.isFile() && shouldDeleteLogFile(entry.name, now))
    .map((entry) => rm(path.join(LOG_DIR, entry.name), { force: true })));
}

function formatTerminalLine(date: Date, entry: ProxyLogEntry): string {
  const time = new Intl.DateTimeFormat("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);

  const statusColor = entry.status === "info"
    ? ANSI.green
    : entry.status === "warning"
      ? ANSI.yellow
      : ANSI.red;
  const model = entry.model ?? "unknown-model";
  const statusCode = typeof entry.statusCode === "number" ? ` ${entry.statusCode}` : "";
  const protocol = entry.protocol ? ` ${ANSI.yellow}${entry.protocol}${ANSI.reset}` : "";
  const route = entry.route ? ` ${ANSI.dim}${entry.route}${ANSI.reset}` : "";
  const duration = typeof entry.durationMs === "number" ? ` ${ANSI.dim}${entry.durationMs}ms${ANSI.reset}` : "";
  const message = entry.message ? ` ${ANSI.dim}${truncate(entry.message, 120)}${ANSI.reset}` : "";

  return `${ANSI.dim}${time}${ANSI.reset} ${statusColor}${entry.status.toUpperCase()}${ANSI.reset} ${ANSI.cyan}${model}${ANSI.reset}${statusCode}${protocol}${route}${duration}${message}`;
}

function formatFileLine(date: Date, entry: ProxyLogEntry): string {
  const parts = [
    date.toISOString(),
    entry.status.toUpperCase(),
    entry.model ?? "unknown-model",
  ];

  if (typeof entry.statusCode === "number") {
    parts.push(String(entry.statusCode));
  }

  if (entry.protocol) {
    parts.push(entry.protocol);
  }

  if (entry.route) {
    parts.push(entry.route);
  }

  if (typeof entry.durationMs === "number") {
    parts.push(`${entry.durationMs}ms`);
  }

  if (entry.message) {
    parts.push(truncate(entry.message.replace(/\s+/g, " "), 200));
  }

  return parts.join(" | ");
}

function formatDate(date: Date): string {
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function parseLogDate(value: string): Date | undefined {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    return undefined;
  }

  const date = new Date(`${value}T00:00:00.000Z`);
  return Number.isNaN(date.getTime()) ? undefined : date;
}

function diffInUtcDays(older: Date, newer: Date): number {
  const olderUtc = Date.UTC(older.getUTCFullYear(), older.getUTCMonth(), older.getUTCDate());
  const newerUtc = Date.UTC(newer.getUTCFullYear(), newer.getUTCMonth(), newer.getUTCDate());
  return Math.floor((newerUtc - olderUtc) / 86_400_000);
}

function truncate(value: string, maxLength: number): string {
  return value.length <= maxLength ? value : `${value.slice(0, maxLength - 1)}...`;
}