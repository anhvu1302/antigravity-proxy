import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm"],
  platform: "node",
  target: "node24",
  outDir: "dist",
  clean: true,
  bundle: true,
  minify: true,
  sourcemap: false,
  dts: false,
  splitting: false,
  shims: false,
});
