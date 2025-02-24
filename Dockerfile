# 构建阶段
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# 运行时阶段
FROM node:18-alpine

# 安装 Chromium 和必要依赖
RUN apk add --no-cache \
    chromium \
    nss \
    ca-certificates \
    && rm -rf /var/cache/apk/*

ENV CHROME_BIN=/usr/bin/chromium-browser

WORKDIR /app

# 只复制必要的文件和生产依赖
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
COPY --from=builder /app/index.js ./
# 如果有其他必要文件，按需添加 COPY

EXPOSE 3000

CMD ["node"， "index.js"]
