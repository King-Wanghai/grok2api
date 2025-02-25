import express from 'express';
import fetch from 'node-fetch';
import FormData from 'form-data';
import dotenv from 'dotenv';
import cors from 'cors';
import puppeteer from 'puppeteer-extra'
import StealthPlugin from 'puppeteer-extra-plugin-stealth'
import { v4 as uuidv4 } from 'uuid';
import Logger from './logger.js';
import { HttpsProxyAgent } from 'https-proxy-agent'; // 使用命名导出

dotenv.config();

// 配置常量
const CONFIG = {
    MODELS: {
        'grok-2': 'grok-latest',
        'grok-2-imageGen': 'grok-latest',
        'grok-2-search': 'grok-latest',
        "grok-3": "grok-3",
        "grok-3-search": "grok-3",
        "grok-3-imageGen": "grok-3",
        "grok-3-deepsearch": "grok-3",
        "grok-3-reasoning": "grok-3"
    },
    API: {
        IS_CUSTOM_SSO: process.env.IS_CUSTOM_SSO === 'true',
        BASE_URL: "https://grok.com",
        API_KEY: process.env.API_KEY || "sk-123456",
        SIGNATURE_COOKIE: null,
        TEMP_COOKIE: null,
        PICGO_KEY: process.env.PICGO_KEY || null, //想要流式生图的话需要填入这个PICGO图床的key
        PICUI_KEY: process.env.PICUI_KEY || null //想要流式生图的话需要填入这个PICUI图床的key 两个图床二选一，默认使用PICGO
    },
    PROXY: process.env.PROXY || '', // 单一代理参数
    SERVER: {
        PORT: process.env.PORT || 3000,
        BODY_LIMIT: '5mb'
    },
    RETRY: {
        MAX_ATTEMPTS: 2//重试次数
    },
    SHOW_THINKING: process.env.SHOW_THINKING === 'true',
    IS_THINKING: false,
    IS_IMG_GEN: false,
    IS_IMG_GEN2: false,
    SSO_INDEX: 0,//sso的索引
    ISSHOW_SEARCH_RESULTS: process.env.ISSHOW_SEARCH_RESULTS === 'true',//是否显示搜索结果
    CHROME_PATH: process.env.CHROME_PATH || "/usr/bin/chromium"//chrome路径
};
puppeteer.use(StealthPlugin())
// 请求头配置
const DEFAULT_HEADERS = {
    'accept': '*/*',
    'accept-language': 'zh-CN,zh;q=0.9',
    'accept-encoding': 'gzip, deflate, br, zstd',
    'content-type': 'text/plain;charset=UTF-8',
    'Connection': 'keep-alive',
    'origin': 'https://grok.com',
    'priority': 'u=1, i',
    'sec-ch-ua': '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
    'baggage': 'sentry-public_key=b311e0f2690c81f25e2c4cf6d4f7ce1c'
};

const proxyAgent = CONFIG.PROXY ? new HttpsProxyAgent(CONFIG.PROXY) : null;

async function initialization() {
    if (CONFIG.API.IS_CUSTOM_SSO) {
        await Utils.get_signature()
        return;
    }
    const ssoArray = process.env.SSO.split(',');
    const ssorwArray = process.env.SSO_RW.split(',');
    ssoArray.forEach((sso, index) => {
        tokenManager.addToken(`sso-rw=${ssorwArray[index]};sso=${sso}`);
    });
    console.log(JSON.stringify(tokenManager.getActiveTokens(), null, 2));
    await Utils.get_signature()
    Logger.info("初始化完成", 'Server');
}


class AuthTokenManager {
    constructor() {
        this.activeTokens = [];
        this.expiredTokens = new Map();
        this.tokenModelFrequency = new Map();
        this.modelRateLimit = {
            "grok-3": { RequestFrequency: 20 },
            "grok-3-deepsearch": { RequestFrequency: 5 },
            "grok-3-reasoning": { RequestFrequency: 5 }
        };
    }

    addToken(token) {
        if (!this.activeTokens.includes(token)) {
            this.activeTokens.push(token);
            this.tokenModelFrequency.set(token, {
                "grok-3": 0,
                "grok-3-deepsearch": 0,
                "grok-3-reasoning": 0
            });
        }
    }
    setToken(token) {
        this.activeTokens = [token];
        this.tokenModelFrequency.set(token, {
            "grok-3": 0,
            "grok-3-deepsearch": 0,
            "grok-3-reasoning": 0
        });
    }

    getTokenByIndex(index, model) {
        if (this.activeTokens.length === 0) {
            return null;
        }
        const token = this.activeTokens[index];
        this.recordModelRequest(token, model);
        return token;
    }

    recordModelRequest(token, model) {
        if (model === 'grok-3-search' || model === 'grok-3-imageGen') {
            model = 'grok-3';
        }

        if (!this.modelRateLimit[model]) return;
        const tokenFrequency = this.tokenModelFrequency.get(token);
        if (tokenFrequency && tokenFrequency[model] !== undefined) {
            tokenFrequency[model]++;
        }
        this.checkAndRemoveTokenIfLimitReached(token);
    }
    setModelLimit(index, model) {
        if (model === 'grok-3-search' || model === 'grok-3-imageGen') {
            model = 'grok-3';
        }
        if (!this.modelRateLimit[model]) return;
        const tokenFrequency = this.tokenModelFrequency.get(this.activeTokens[index]);
        tokenFrequency[model] = 9999;
    }
    isTokenModelLimitReached(index, model) {
        if (model === 'grok-3-search' || model === 'grok-3-imageGen') {
            model = 'grok-3';
        }
        if (!this.modelRateLimit[model]) return;
        const token = this.activeTokens[index];
        const tokenFrequency = this.tokenModelFrequency.get(token);

        if (!tokenFrequency) {
            return false;
        }
        return tokenFrequency[model] >= this.modelRateLimit[model].RequestFrequency;
    }
    checkAndRemoveTokenIfLimitReached(token) {
        const tokenFrequency = this.tokenModelFrequency.get(token);
        if (!tokenFrequency) return;

        const isLimitReached = Object.keys(tokenFrequency).every(model =>
            tokenFrequency[model] >= this.modelRateLimit[model].RequestFrequency
        );

        if (isLimitReached) {
            const tokenIndex = this.activeTokens.indexOf(token);
            if (tokenIndex !== -1) {
                this.removeTokenByIndex(tokenIndex);
            }
        }
    }

    removeTokenByIndex(index) {
        if (!this.isRecoveryProcess) {
            this.startTokenRecoveryProcess();
        }
        const token = this.activeTokens[index];
        this.expiredTokens.set(token, Date.now());
        this.activeTokens.splice(index, 1);
        this.tokenModelFrequency.delete(token);
        Logger.info(`令牌${token}已达到上限，已移除`, 'TokenManager');
    }

    startTokenRecoveryProcess() {
        if (CONFIG.API.IS_CUSTOM_SSO) return;
        setInterval(() => {
            const now = Date.now();
            for (const [token, expiredTime] of this.expiredTokens.entries()) {
                if (now - expiredTime >= 2 * 60 * 60 * 1000) {
                    this.tokenModelFrequency.set(token, {
                        "grok-3": 0,
                        "grok-3-deepsearch": 0,
                        "grok-3-reasoning": 0
                    });
                    this.activeTokens.push(token);
                    this.expiredTokens.delete(token);
                    Logger.info(`令牌${token}已恢复，已添加到可用令牌列表`, 'TokenManager');
                }
            }
        }, 2 * 60 * 60 * 1000);
    }

    getTokenCount() {
        return this.activeTokens.length || 0;
    }

    getActiveTokens() {
        return [...this.activeTokens];
    }
}

class Utils {
    static async extractGrokHeaders() {
      Logger.info("开始提取头信息", 'Server');
      try {
        // 确保 proxyArgs 在这里定义
        const proxyArgs = CONFIG.PROXY ? [`--proxy-server=${CONFIG.PROXY}`] : [];
        const browser = await puppeteer.launch({
          headless: true,
          args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
            ...proxyArgs // 使用正确定义的 proxyArgs
          ],
          executablePath: CONFIG.CHROME_PATH
        });

        const page = await browser.newPage();
        await page.goto('https://grok.com/', { waitUntil: 'domcontentloaded' });
        await page.evaluate(() => {
          return new Promise(resolve => setTimeout(resolve, 5000));
        });
        const cookies = await page.cookies();
        const targetHeaders = ['x-anonuserid', 'x-challenge', 'x-signature'];
        const extractedHeaders = {};
        for (const cookie of cookies) {
          if (targetHeaders.includes(cookie.name.toLowerCase())) {
            extractedHeaders[cookie.name.toLowerCase()] = cookie.value;
          }
        }
        await browser.close();
        Logger.info('提取的头信息:', JSON.stringify(extractedHeaders, null, 2), 'Server');
        return extractedHeaders;
      } catch (error) {
        Logger.error('获取头信息出错:', error, 'Server');
        return null;
      }
    }
    static async get_signature() {
        if (CONFIG.API.TEMP_COOKIE) {
            return CONFIG.API.TEMP_COOKIE;
        }
        Logger.info("刷新认证信息", 'Server');
        let retryCount = 0;
        while (!CONFIG.API.TEMP_COOKIE || retryCount < CONFIG.RETRY.MAX_ATTEMPTS) {
            let headers = await Utils.extractGrokHeaders();
            if (headers) {
                Logger.info("获取认证信息成功", 'Server');
                CONFIG.API.TEMP_COOKIE = { cookie: `x-anonuserid=${headers["x-anonuserid"]}; x-challenge=${headers["x-challenge"]}; x-signature=${headers["x-signature"]}` };
                return;
            }
            retryCount++;
            if (retryCount >= CONFIG.RETRY.MAX_ATTEMPTS) {
                throw new Error(`获取认证信息失败!`);
            }
        }
    }
    static async organizeSearchResults(searchResults) {
        // 确保传入的是有效的搜索结果对象
        if (!searchResults || !searchResults.results) {
            return '';
        }

        const results = searchResults.results;
        const formattedResults = results.map((result, index) => {
            // 处理可能为空的字段
            const title = result.title || '未知标题';
            const url = result.url || '#';
            const preview = result.preview || '无预览内容';

            return `\r\n<details><summary>资料[${index}]: ${title}</summary>\r\n${preview}\r\n\n[Link](${url})\r\n</details>`;
        });
        return formattedResults.join('\n\n');
    }
    static async createAuthHeaders(model) {
        return {
            'cookie': `${await tokenManager.getTokenByIndex(CONFIG.SSO_INDEX, model)}`
        };
    }
}


class GrokApiClient {
    constructor(modelId) {
        if (!CONFIG.MODELS[modelId]) {
            throw new Error(`不支持的模型: ${modelId}`);
        }
        this.modelId = CONFIG.MODELS[modelId];
    }

    processMessageContent(content) {
        if (typeof content === 'string') return content;
        return null;
    }
    // 获取图片类型
    getImageType(base64String) {
        let mimeType = 'image/jpeg';
        if (base64String.includes('data:image')) {
            const matches = base64String.match(/data:([a-zA-Z0-9]+\/[a-zA-Z0-9-.+]+);base64,/);
            if (matches) {
                mimeType = matches[1];
            }
        }
        const extension = mimeType.split('/')[1];
        const fileName = `image.${extension}`;

        return {
            mimeType: mimeType,
            fileName: fileName
        };
    }

    async uploadBase64Image(base64Data, url) {
        try {
            // 处理 base64 数据
            let imageBuffer;
            if (base64Data.includes('data:image')) {
                imageBuffer = base64Data.split(',')[1];
            } else {
                imageBuffer = base64Data
            }
            const { mimeType, fileName } = this.getImageType(base64Data);
            let uploadData = {
                rpc: "uploadFile",
                req: {
                    fileName: fileName,
                    fileMimeType: mimeType,
                    content: imageBuffer
                }
            };
            Logger.info("发送图片请求", 'Server');
            // 发送请求
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    ...CONFIG.DEFAULT_HEADERS,
                    ...CONFIG.API.SIGNATURE_COOKIE
                },
                body: JSON.stringify(uploadData),
                agent: proxyAgent // 使用单一代理
            });

            if (!response.ok) {
                Logger.error(`上传图片失败,状态码:${response.status},原因:${response.error}`, 'Server');
                return '';
            }

            const result = await response.json();
            Logger.info('上传图片成功:', result, 'Server');
            return result.fileMetadataId;

        } catch (error) {
            Logger.error(error, 'Server');
            return '';
        }
    }

    async prepareChatRequest(request) {
        if ((request.model === 'grok-2-imageGen' || request.model === 'grok-3-imageGen') && !CONFIG.API.PICGO_KEY && !CONFIG.API.PICUI_KEY && request.stream) {
            throw new Error(`该模型流式输出需要配置PICGO或者PICUI图床密钥!`);
        }

        // 处理画图模型的消息限制
        let todoMessages = request.messages;
        if (request.model === 'grok-2-imageGen' || request.model === 'grok-3-imageGen') {
            const lastMessage = todoMessages[todoMessages.length - 1];
            if (lastMessage.role !== 'user') {
                throw new Error('画图模型的最后一条消息必须是用户消息!');
            }
            todoMessages = [lastMessage];
        }

        const fileAttachments = [];
        let messages = '';
        let lastRole = null;
        let lastContent = '';
        const search = request.model === 'grok-2-search' || request.model === 'grok-3-search';

        // 移除<think>标签及其内容和base64图片
        const removeThinkTags = (text) => {
            text = text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
            text = text.replace(/!\[image\]\(data:.*?base64,.*?\)/g, '[图片]');
            return text;
        };

        const processImageUrl = async (content) => {
            if (content.type === 'image_url' && content.image_url.url.includes('data:image')) {
                const imageResponse = await this.uploadBase64Image(
                    content.image_url.url,
                    `${CONFIG.API.BASE_URL}/api/rpc`
                );
                return imageResponse;
            }
            return null;
        };

        const processContent = async (content) => {
            if (Array.isArray(content)) {
                let textContent = '';
                for (const item of content) {
                    if (item.type === 'image_url') {
                        textContent += (textContent ? '\n' : '') + "[图片]";
                    } else if (item.type === 'text') {
                        textContent += (textContent ? '\n' : '') + removeThinkTags(item.text);
                    }
                }
                return textContent;
            } else if (typeof content === 'object' && content !== null) {
                if (content.type === 'image_url') {
                    return "[图片]";
                } else if (content.type === 'text') {
                    return removeThinkTags(content.text);
                }
            }
            return removeThinkTags(this.processMessageContent(content));
        };

        for (const current of todoMessages) {
            const role = current.role === 'assistant' ? 'assistant' : 'user';
            const isLastMessage = current === todoMessages[todoMessages.length - 1];

            // 处理图片附件
            if (isLastMessage && current.content) {
                if (Array.isArray(current.content)) {
                    for (const item of current.content) {
                        if (item.type === 'image_url') {
                            const processedImage = await processImageUrl(item);
                            if (processedImage) fileAttachments.push(processedImage);
                        }
                    }
                } else if (current.content.type === 'image_url') {
                    const processedImage = await processImageUrl(current.content);
                    if (processedImage) fileAttachments.push(processedImage);
                }
            }

            // 处理文本内容
            const textContent = await processContent(current.content);

            if (textContent || (isLastMessage && fileAttachments.length > 0)) {
                if (role === lastRole && textContent) {
                    lastContent += '\n' + textContent;
                    messages = messages.substring(0, messages.lastIndexOf(`${role.toUpperCase()}: `)) +
                        `${role.toUpperCase()}: ${lastContent}\n`;
                } else {
                    messages += `${role.toUpperCase()}: ${textContent || '[图片]'}\n`;
                    lastContent = textContent;
                    lastRole = role;
                }
            }
        }

        return {
            modelName: this.modelId,
            message: messages.trim(),
            fileAttachments: fileAttachments.slice(0, 4),
            imageAttachments: [],
            disableSearch: false,
            enableImageGeneration: true,
            returnImageBytes: false,
            returnRawGrokInXaiRequest: false,
            enableImageStreaming: false,
            imageGenerationCount: 1,
            forceConcise: false,
            toolOverrides: {
                imageGen: request.model === 'grok-2-imageGen' || request.model === 'grok-3-imageGen',
                webSearch: search,
                xSearch: search,
                xMediaSearch: search,
                trendsSearch: search,
                xPostAnalyze: search
            },
            enableSideBySide: true,
            isPreset: false,
            sendFinalMetadata: true,
            customInstructions: "",
            deepsearchPreset: request.model === 'grok-3-deepsearch' ? "default" : "",
            isReasoning: request.model === 'grok-3-reasoning'
        };
    }
}

class MessageProcessor {
    static createChatResponse(message, model, isStream = false) {
        const baseResponse = {
            id: `chatcmpl-${uuidv4()}`,
            created: Math.floor(Date.now() / 1000),
            model: model
        };

        if (isStream) {
            return {
                ...baseResponse,
                object: 'chat.completion.chunk',
                choices: [{
                    index: 0,
                    delta: {
                        content: message
                    }
                }]
            };
        }

        return {
            ...baseResponse,
            object: 'chat.completion',
            choices: [{
                index: 0,
                message: {
                    role: 'assistant',
                    content: message
                },
                finish_reason: 'stop'
            }],
            usage: null
        };
    }
}
async function processModelResponse(linejosn, model) {
    let result = { token: null, imageUrl: null }
    if (CONFIG.IS_IMG_GEN) {
        if (linejosn?.cachedImageGenerationResponse && !CONFIG.IS_IMG_GEN2) {
            result.imageUrl = linejosn.cachedImageGenerationResponse.imageUrl;
        }
        return result;
    }

    //非生图模型的处理
    switch (model) {
        case 'grok-2':
            result.token = linejosn?.token;
            return result;
        case 'grok-2-search':
        case 'grok-3-search':
            if (linejosn?.webSearchResults && CONFIG.ISSHOW_SEARCH_RESULTS) {
                result.token = `\r\n<think>${await Utils.organizeSearchResults(linejosn.webSearchResults)}</think>\r\n`;
            } else {
                result.token = linejosn?.token;
            }
            return result;
        case 'grok-3':
            result.token = linejosn?.token;
            return result;
        case 'grok-3-deepsearch':
            if (linejosn.messageTag === "final") {
                result.token = linejosn?.token;
            }
            return result;
        case 'grok-3-reasoning':
            if (linejosn?.isThinking && !CONFIG.SHOW_THINKING) return result;

            if (linejosn?.isThinking && !CONFIG.IS_THINKING) {
                result.token = "<think>" + linejosn?.token;
                CONFIG.IS_THINKING = true;
            } else if (!linejosn.isThinking && CONFIG.IS_THINKING) {
                result.token = "</think>" + linejosn?.token;
                CONFIG.IS_THINKING = false;
            } else {
                result.token = linejosn?.token;
            }
            return result;
    }
    return result;
}

async function handleResponse(response, model, res, isStream) {
    try {
        const stream = response.body;
        let buffer = '';
        let fullResponse = '';
        const dataPromises = [];
        if (isStream) {
            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');
        }
        Logger.info("开始处理流式响应", 'Server');

        return new Promise((resolve, reject) => {
            stream.on('data', async (chunk) => {
                buffer += chunk.toString();
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.trim()) continue;
                    const trimmedLine = line.trim();
                    if (trimmedLine.startsWith('data: ')) {
                        const data = trimmedLine.substring(6);
                        try {
                            if (!data.trim()) continue;
                            if (data === "[DONE]") continue;                     
                            const linejosn = JSON.parse(data);
                            if (linejosn?.error) {                          
                                Logger.error(JSON.stringify(linejosn, null, 2), 'Server');
                                if(linejosn.error?.name === "RateLimitError"){
                                    CONFIG.API.TEMP_COOKIE = null;
                                }
                                stream.destroy();
                                reject(new Error("RateLimitError"));
                                return;
                            }
                            if (linejosn?.doImgGen || linejosn?.imageAttachmentInfo) {
                                CONFIG.IS_IMG_GEN = true;
                            }
                            const processPromise = (async () => {
                                const result = await processModelResponse(linejosn, model);

                                if (result.token) {
                                    if (isStream) {
                                        res.write(`data: ${JSON.stringify(MessageProcessor.createChatResponse(result.token, model, true))}\n\n`);
                                    } else {
                                        fullResponse += result.token;
                                    }
                                }
                                if (result.imageUrl) {
                                    CONFIG.IS_IMG_GEN2 = true;
                                    const dataImage = await handleImageResponse(result.imageUrl);
                                    if (isStream) {
                                        res.write(`data: ${JSON.stringify(MessageProcessor.createChatResponse(dataImage, model, true))}\n\n`);
                                    } else {
                                        res.json(MessageProcessor.createChatResponse(dataImage, model));
                                    }
                                }
                            })();
                            dataPromises.push(processPromise);
                        } catch (error) {
                            Logger.error(error, 'Server');
                            continue;
                        }
                    }
                }
            });

            stream.on('end', async () => {
                try {
                    await Promise.all(dataPromises);
                    if (isStream) {
                        res.write('data: [DONE]\n\n');
                        res.end();
                    } else {
                        if (!CONFIG.IS_IMG_GEN2) {
                            res.json(MessageProcessor.createChatResponse(fullResponse, model));
                        }
                    }
                    CONFIG.IS_IMG_GEN = false;
                    CONFIG.IS_IMG_GEN2 = false;
                    resolve();
                } catch (error) {
                    Logger.error(error, 'Server');
                    reject(error);
                }
            });

            stream.on('error', (error) => {
                Logger.error(error, 'Server');
                reject(error);
            });
        });
    } catch (error) {
        Logger.error(error, 'Server');
        CONFIG.IS_IMG_GEN = false;
        CONFIG.IS_IMG_GEN2 = false;
        throw new Error(error);
    }
}

async function handleImageResponse(imageUrl) {
    const MAX_RETRIES = 2;
    let retryCount = 0;
    let imageBase64Response;

    while (retryCount < MAX_RETRIES) {
        try {
            imageBase64Response = await fetch(`https://assets.grok.com/${imageUrl}`, {
                method: 'GET',
                headers: {
                    ...DEFAULT_HEADERS,
                    ...CONFIG.API.SIGNATURE_COOKIE
                }
            });

            if (imageBase64Response.ok) break;
            retryCount++;
            if (retryCount === MAX_RETRIES) {
                throw new Error(`上游服务请求失败! status: ${imageBase64Response.status}`);
            }
            await new Promise(resolve => setTimeout(resolve, CONFIG.API.RETRY_TIME * retryCount));

        } catch (error) {
            Logger.error(error, 'Server');
            retryCount++;
            if (retryCount === MAX_RETRIES) {
                throw error;
            }
            await new Promise(resolve => setTimeout(resolve, CONFIG.API.RETRY_TIME * retryCount));
        }
    }


    const arrayBuffer = await imageBase64Response.arrayBuffer();
    const imageBuffer = Buffer.from(arrayBuffer);

    if (!CONFIG.API.PICGO_KEY && !CONFIG.API.PICUI_KEY) {
        const base64Image = imageBuffer.toString('base64');
        const imageContentType = imageBase64Response.headers.get('content-type');
        return `![image](data:${imageContentType};base64,${base64Image})`
    }

    const formData = new FormData();
    if(CONFIG.API.PICGO_KEY){
        formData.append('source', imageBuffer, {
            filename: `image-${Date.now()}.jpg`,
            contentType: 'image/jpeg'
        });
        const formDataHeaders = formData.getHeaders();
        const responseURL = await fetch("https://www.picgo.net/api/1/upload", {
            method: "POST",
            headers: {
            ...formDataHeaders,
            "Content-Type": "multipart/form-data",
            "X-API-Key": CONFIG.API.PICGO_KEY
        },
        body: formData
        });
        if (!responseURL.ok) {
            return "生图失败，请查看图床密钥是否设置正确"
        } else {
            Logger.info("生图成功", 'Server');
            const result = await responseURL.json();
            return `![image](${result.image.url})`
        }
    }else if(CONFIG.API.PICUI_KEY){
        const formData = new FormData();
        formData.append('file', imageBuffer, {
            filename: `image-${Date.now()}.jpg`,
            contentType: 'image/jpeg'
        });
        const formDataHeaders = formData.getHeaders();
        const responseURL = await fetch("https://picui.cn/api/v1/upload", {
            method: "POST",
            headers: {
                ...formDataHeaders,
                "Accept": "application/json",
                'Authorization': `Bearer ${CONFIG.API.PICUI_KEY}`
            },
            body: formData
        });
        if (!responseURL.ok) {
            return "生图失败，请查看图床密钥是否设置正确"
        } else {
            Logger.info("生图成功", 'Server');
            const result = await responseURL.json();
            return `![image](${result.data.links.url})`
        }
    }
}

const tokenManager = new AuthTokenManager();
await initialization();

// 中间件配置
const app = express();
app.use(Logger.requestLogger);
app.use(express.json({ limit: CONFIG.SERVER.BODY_LIMIT }));
app.use(express.urlencoded({ extended: true, limit: CONFIG.SERVER.BODY_LIMIT }));
app.use(cors({
    origin: '*',
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));

app.get('/v1/models', (req, res) => {
    res.json({
        object: "list",
        data: Object.keys(CONFIG.MODELS).map((model, index) => ({
            id: model,
            object: "model",
            created: Math.floor(Date.now() / 1000),
            owned_by: "grok",
        }))
    });
});


app.post('/v1/chat/completions', async (req, res) => {
    try {
        const authToken = req.headers.authorization?.replace('Bearer ', '');
        if (CONFIG.API.IS_CUSTOM_SSO) {
            if (authToken && authToken.includes(';')) {
                const parts = authToken.split(';');
                const result = [
                    `sso=${parts[0]}`,
                    `ssp_rw=${parts[1]}`
                ].join(';');
                tokenManager.setToken(result);
            } else {
                return res.status(401).json({ error: '自定义的SSO令牌格式错误' });
            }
        } else if (authToken !== CONFIG.API.API_KEY) {
            return res.status(401).json({ error: 'Unauthorized' });
        }
        let isTempCookie = req.body.model.includes("grok-2");
        let retryCount = 0;
        const grokClient = new GrokApiClient(req.body.model);
        const requestPayload = await grokClient.prepareChatRequest(req.body);
        Logger.info(`请求体: ${JSON.stringify(requestPayload,null,2)}`, 'Server');

        while (retryCount < CONFIG.RETRY.MAX_ATTEMPTS) {
            retryCount++;
            if (!CONFIG.API.TEMP_COOKIE) {
                await Utils.get_signature();
            }

            if (isTempCookie) {
                CONFIG.API.SIGNATURE_COOKIE = CONFIG.API.TEMP_COOKIE;
                Logger.info(`已切换为临时令牌`, 'Server');
            } else {
                CONFIG.API.SIGNATURE_COOKIE = await Utils.createAuthHeaders(req.body.model);
            }
            if (!CONFIG.API.SIGNATURE_COOKIE) {
                throw new Error('无可用令牌');
            }
            Logger.info(`当前令牌索引: ${CONFIG.SSO_INDEX}`, 'Server');
            Logger.info(`当前令牌: ${JSON.stringify(CONFIG.API.SIGNATURE_COOKIE,null,2)}`, 'Server');
            const newMessageReq = await fetch(`${CONFIG.API.BASE_URL}/api/rpc`, {
                method: 'POST',
                headers: {
                    ...DEFAULT_HEADERS,
                    ...CONFIG.API.SIGNATURE_COOKIE
                },
                body: JSON.stringify({
                    rpc: "createConversation",
                    req: {
                        temporary: false
                    }
                }),
                agent: proxyAgent
            });
            let conversationId;
            var responseText2 = await newMessageReq.clone().text();
            if (newMessageReq.status === 200) {
                const responseText = await newMessageReq.json();
                conversationId = responseText.conversationId;
            } else {
                Logger.error(`创建会话请求: ${responseText2}`, 'Server');
                throw new Error(`创建会话响应错误: ${responseText2}`);
            }

            const response = await fetch(`${CONFIG.API.BASE_URL}/api/conversations/${conversationId}/responses`, {
                method: 'POST',
                headers: {
                    "accept": "text/event-stream",
                    "baggage": "sentry-public_key=b311e0f2690c81f25e2c4cf6d4f7ce1c",
                    "content-type": "text/plain;charset=UTF-8",
                    "Connection": "keep-alive",
                    ...CONFIG.API.SIGNATURE_COOKIE
                },
                body: JSON.stringify(requestPayload),
                agent: proxyAgent
            });

            if (response.ok) {
                Logger.info(`请求成功`, 'Server');
                CONFIG.SSO_INDEX = (CONFIG.SSO_INDEX + 1) % tokenManager.getTokenCount();
                Logger.info(`当前剩余可用令牌数: ${tokenManager.getTokenCount()}`, 'Server');
                try {
                    await handleResponse(response, req.body.model, res, req.body.stream);
                    Logger.info(`请求结束`, 'Server');
                    return;
                } catch (error) {
                    Logger.error(error, 'Server');
                    if (isTempCookie) {
                        await Utils.get_signature();
                    } else {
                        tokenManager.setModelLimit(CONFIG.SSO_INDEX, req.body.model);
                        for (let i = 1; i <= tokenManager.getTokenCount(); i++) {
                            CONFIG.SSO_INDEX = (CONFIG.SSO_INDEX + 1) % tokenManager.getTokenCount();
                            if (!tokenManager.isTokenModelLimitReached(CONFIG.SSO_INDEX, req.body.model)) {
                                break;
                            } else if (i >= tokenManager.getTokenCount()) {
                                throw new Error(`${req.body.model} 次数已达上限，请切换其他模型或者重新对话`);
                            }
                        }
                    }
                }
            } else {
                if (response.status === 429) {
                    if (isTempCookie) {
                        await Utils.get_signature();
                    } else {                      
                        tokenManager.setModelLimit(CONFIG.SSO_INDEX, req.body.model);
                        for (let i = 1; i <= tokenManager.getTokenCount(); i++) {
                            CONFIG.SSO_INDEX = (CONFIG.SSO_INDEX + 1) % tokenManager.getTokenCount();
                            if (!tokenManager.isTokenModelLimitReached(CONFIG.SSO_INDEX, req.body.model)) {
                                break;
                            } else if (i >= tokenManager.getTokenCount()) {
                                throw new Error(`${req.body.model} 次数已达上限，请切换其他模型或者重新对话`);
                            }
                        }
                    }
                } else {                
                    // 非429错误直接抛出
                    if (isTempCookie) {
                        await Utils.get_signature();
                    } else {
                        Logger.error(`令牌异常错误状态!status: ${response.status}， 已移除当前令牌${CONFIG.SSO_INDEX.cookie}`, 'Server');
                        tokenManager.removeTokenByIndex(CONFIG.SSO_INDEX);
                        Logger.info(`当前剩余可用令牌数: ${tokenManager.getTokenCount()}`, 'Server');
                        CONFIG.SSO_INDEX = (CONFIG.SSO_INDEX + 1) % tokenManager.getTokenCount();
                    }
                }
            }
        }
        throw new Error('当前模型所有令牌都已耗尽');
    } catch (error) {
        Logger.error(error, 'ChatAPI');
        res.status(500).json({
            error: {
                message: error.message || error,
                type: 'server_error'
            }
        });
    }
});


app.use((req, res) => {
    res.status(200).send('api运行正常');
});


app.listen(CONFIG.SERVER.PORT, () => {
    Logger.info(`服务器已启动，监听端口: ${CONFIG.SERVER.PORT}`, 'Server');
});
