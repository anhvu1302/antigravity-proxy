module.exports = {
  apps: [
    {
      name: "gemini-proxy",
      cwd: "/Users/VanAnh/WorkSpace/Personal/gemini-proxy",
      script: "dist/index.js",
      interpreter: "/Users/VanAnh/.nvm/versions/node/v24.11.1/bin/node",
      autorestart: true,
      restart_delay: 2000,
      env: {
        NODE_ENV: "production",
      },
    },
  ],
};