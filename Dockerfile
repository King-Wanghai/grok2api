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

# 通用代理环境变量（默认空，运行时注入）
ENV PROXY=""
# 初始化标准代理变量
ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""
ENV ALL_PROXY=""
ENV NO_PROXY="localhost,127.0.0.1"

# 添加一个入口脚本
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /app

COPY package*.json ./
RUN npm install --production

COPY . .

EXPOSE 3000

# 使用入口脚本启动应用
ENTRYPOINT ["entrypoint.sh"]
CMD ["node", "index.js"]
