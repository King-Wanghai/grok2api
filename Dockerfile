# 使用官方 Python 3.10 镜像作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装所需的 Python 依赖
RUN pip install --no-cache-dir flask requests curl_cffi werkzeug loguru

# 复制应用代码到容器
COPY . /app/

# 设置环境变量
ENV PORT=3000

# 开放容器的 3000 端口
EXPOSE 3000

# 启动 Flask 应用
CMD ["python", "app.py"]
