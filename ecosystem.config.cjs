module.exports = {
  apps: [
    {
      name: "antigravity-proxy",
      cwd: "/Users/VanAnh/WorkSpace/Personal/antigravity-proxy",
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