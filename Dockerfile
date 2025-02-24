FROM node:18-alpine

# 安装必要的工具和 Chromium
RUN apk add --no-cache \
    wget \
    chromium \
    nss \
    ca-certificates \
    && rm -rf /var/cache/apk/*

# 设置 Chromium 可执行文件路径
ENV CHROME_BIN=/usr/bin/chromium-browser

# 代理环境变量（默认空，运行时注入）
ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""
ENV NO_PROXY="localhost,127.0.0.1"

WORKDIR /app

COPY package*.json ./
RUN npm install --production

COPY . .

EXPOSE 3000

CMD ["node", "index.js"]
