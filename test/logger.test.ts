import assert from "node:assert/strict";
import test from "node:test";

import { getLogFilePath, shouldDeleteLogFile } from "../src/logger.js";

test("builds log file paths by UTC date", () => {
  const filePath = getLogFilePath(new Date("2026-06-12T08:30:00.000Z"));

  assert.match(filePath, /logs\/proxy-2026-06-12\.log$/);
});

test("keeps log files newer than seven days", () => {
  const now = new Date("2026-06-12T12:00:00.000Z");

  assert.equal(shouldDeleteLogFile("proxy-2026-06-06.log", now), false);
});

test("deletes log files at or beyond seven days", () => {
  const now = new Date("2026-06-12T12:00:00.000Z");

  assert.equal(shouldDeleteLogFile("proxy-2026-06-05.log", now), true);
  assert.equal(shouldDeleteLogFile("notes.txt", now), false);
});