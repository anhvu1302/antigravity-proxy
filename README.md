# antigravity-proxy Setup Guide

This document explains how to run the project locally and keep it running with PM2 after reboot.

## 1. Prerequisites

- Node.js 24.x
- pnpm 11+
- PM2 (optional, for background process management)

Install PM2 globally if needed:

```bash
npm install -g pm2
```

## 2. Install Dependencies

From the project root:

```bash
pnpm install
```

## 3. Environment Configuration

Copy the example file:

```bash
cp .env.example .env
```

Update `.env` values as needed:

- `HOST` (default: `127.0.0.1`)
- `PORT` (default: `8045`)
- `PROXY_FORWARD_BASE_URL` (optional compatibility forward mode)
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` (used if request headers do not provide API key)

Notes:

- API key can also be sent via headers: `x-goog-api-key`, `x-api-key`, or `Authorization: Bearer <token>`.
- `GEMINI_BASE_URL` is optional and defaults to `https://generativelanguage.googleapis.com/v1beta`.

## 4. Run in Development

```bash
pnpm dev
```

## 5. Build and Run in Production Mode

Build is handled by tsup for Node backend output (bundle + minify).

```bash
pnpm build
pnpm start
```

Health check:

```bash
curl http://127.0.0.1:8045/health
```

## 6. Run with PM2

Start the app from ecosystem config:

```bash
pm2 start ecosystem.config.cjs --only antigravity-proxy
```

Check process:

```bash
pm2 status antigravity-proxy
```

Save process list so PM2 can restore it:

```bash
pm2 save
```

## 7. Auto-start PM2 on macOS Boot

Generate startup command:

```bash
pm2 startup
```

PM2 will print a `sudo ... pm2 startup launchd ...` command. Run that exact command once.

After that, restart your machine and verify:

```bash
pm2 status antigravity-proxy
```

## 8. Useful PM2 Commands

```bash
pm2 logs antigravity-proxy
pm2 restart antigravity-proxy --update-env
pm2 stop antigravity-proxy
pm2 delete antigravity-proxy
pm2 list
```
