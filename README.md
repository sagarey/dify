![cover-v5-optimized](https://github.com/langgenius/dify/assets/13230914/f9e19af5-61ba-4119-b926-d10c4c06ebab)

<p align="center">
  <a href="https://cloud.dify.ai">Dify Cloud</a> ·
  <a href="https://docs.dify.ai/getting-started/install-self-hosted">Self-hosting</a> ·
  <a href="https://docs.dify.ai">Documentation</a> ·
  <a href="https://cal.com/guchenhe/60-min-meeting">Enterprise inquiry</a>
</p>

<p align="center">
    <a href="https://dify.ai" target="_blank">
        <img alt="Static Badge" src="https://img.shields.io/badge/Product-F04438"></a>
    <a href="https://dify.ai/pricing" target="_blank">
        <img alt="Static Badge" src="https://img.shields.io/badge/free-pricing?logo=free&color=%20%23155EEF&label=pricing&labelColor=%20%23528bff"></a>
    <a href="https://discord.gg/FngNHpbcY7" target="_blank">
        <img src="https://img.shields.io/discord/1082486657678311454?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb"
            alt="chat on Discord"></a>
    <a href="https://twitter.com/intent/follow?screen_name=dify_ai" target="_blank">
        <img src="https://img.shields.io/twitter/follow/dify_ai?logo=X&color=%20%23f5f5f5"
            alt="follow on Twitter"></a>
    <a href="https://hub.docker.com/u/langgenius" target="_blank">
        <img alt="Docker Pulls" src="https://img.shields.io/docker/pulls/langgenius/dify-web?labelColor=%20%23FDB062&color=%20%23f79009"></a>
    <a href="https://github.com/langgenius/dify/graphs/commit-activity" target="_blank">
        <img alt="Commits last month" src="https://img.shields.io/github/commit-activity/m/langgenius/dify?labelColor=%20%2332b583&color=%20%2312b76a"></a>
    <a href="https://github.com/langgenius/dify/" target="_blank">
        <img alt="Issues closed" src="https://img.shields.io/github/issues-search?query=repo%3Alanggenius%2Fdify%20is%3Aclosed&label=issues%20closed&labelColor=%20%237d89b0&color=%20%235d6b98"></a>
    <a href="https://github.com/langgenius/dify/discussions/" target="_blank">
        <img alt="Discussion posts" src="https://img.shields.io/github/discussions/langgenius/dify?labelColor=%20%239b8afb&color=%20%237a5af8"></a>
</p>

<p align="center">
  <a href="./README.md"><img alt="README in English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="./README_CN.md"><img alt="简体中文版自述文件" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="./README_JA.md"><img alt="日本語のREADME" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="./README_ES.md"><img alt="README en Español" src="https://img.shields.io/badge/Español-d9d9d9"></a>
  <a href="./README_FR.md"><img alt="README en Français" src="https://img.shields.io/badge/Français-d9d9d9"></a>
  <a href="./README_KL.md"><img alt="README tlhIngan Hol" src="https://img.shields.io/badge/Klingon-d9d9d9"></a>
  <a href="./README_KR.md"><img alt="README in Korean" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
  <a href="./README_AR.md"><img alt="README بالعربية" src="https://img.shields.io/badge/العربية-d9d9d9"></a>
</p>


Dify 是一个开源的大语言模型应用开发平台。其直观的界面结合了 AI 工作流、RAG 管道、智能体功能、模型管理、可观测性等特性，让您能够快速从原型进入生产。以下是核心功能列表：
</br> </br>

**1. 工作流**: 
  在可视化画布上构建和测试强大的 AI 工作流，利用以下所有功能以及更多特性。


  https://github.com/langgenius/dify/assets/13230914/356df23e-1604-483d-80a6-9517ece318aa



**2. 全面的模型支持**: 
  与来自数十家推理供应商和自托管解决方案的数百个专有/开源大语言模型无缝集成，涵盖 GPT、Mistral、Llama3 以及任何兼容 OpenAI API 的模型。受支持的模型供应商完整列表可在[此处](https://docs.dify.ai/getting-started/readme/model-providers)找到。

![providers-v5](https://github.com/langgenius/dify/assets/13230914/5a17bdbe-097a-4100-8363-40255b70f6e3)


**3. 提示词 IDE**: 
  用于制作提示词、比较模型性能以及为基于聊天的应用添加文本转语音等附加功能的直观界面。

**4. RAG 管道**: 
  广泛的 RAG 功能，涵盖从文档摄取到检索的所有环节，开箱即用地支持从 PDF、PPT 和其他常见文档格式中提取文本。

**5. 智能体功能**: 
  您可以基于大语言模型函数调用或 ReAct 定义智能体，并为智能体添加预构建或自定义工具。Dify 为 AI 智能体提供了 50+ 个内置工具，如 Google 搜索、DELL·E、Stable Diffusion 和 WolframAlpha。

**6. LLMOps**: 
  监控和分析应用程序日志和性能。您可以根据生产数据和标注持续改进提示词、数据集和模型。

**7. 后端即服务**: 
  Dify 的所有产品都提供相应的 API，因此您可以轻松地将 Dify 集成到自己的业务逻辑中。


## 功能对比
<table style="width: 100%;">
  <tr>
    <th align="center">功能</th>
    <th align="center">Dify.AI</th>
    <th align="center">LangChain</th>
    <th align="center">Flowise</th>
    <th align="center">OpenAI Assistants API</th>
  </tr>
  <tr>
    <td align="center">编程方式</td>
    <td align="center">API + 应用导向</td>
    <td align="center">Python 代码</td>
    <td align="center">应用导向</td>
    <td align="center">API 导向</td>
  </tr>
  <tr>
    <td align="center">支持的大语言模型</td>
    <td align="center">丰富多样</td>
    <td align="center">丰富多样</td>
    <td align="center">丰富多样</td>
    <td align="center">仅 OpenAI</td>
  </tr>
  <tr>
    <td align="center">RAG 引擎</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
  </tr>
  <tr>
    <td align="center">智能体</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
    <td align="center">✅</td>
  </tr>
  <tr>
    <td align="center">工作流</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
  </tr>
  <tr>
    <td align="center">可观测性</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
    <td align="center">❌</td>
  </tr>
  <tr>
    <td align="center">企业功能（SSO/访问控制）</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
    <td align="center">❌</td>
    <td align="center">❌</td>
  </tr>
  <tr>
    <td align="center">本地部署</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
  </tr>
</table>

## 使用 Dify

- **云服务 </br>**
我们为任何人提供零设置的 [Dify Cloud](https://dify.ai) 服务。它提供自部署版本的所有功能，沙盒计划包含 200 次免费的 GPT-4 调用。

- **自托管 Dify 社区版</br>**
通过此[快速开始指南](#quick-start)在您的环境中快速运行 Dify。
使用我们的[文档](https://docs.dify.ai)获取更多参考和深入指导。

- **企业/组织版 Dify</br>**
我们提供额外的企业级功能。[与我们安排会议](https://cal.com/guchenhe/30min)或[发送邮件](mailto:business@dify.ai?subject=[GitHub]Business%20License%20Inquiry)讨论企业需求。</br>
  > 对于使用 AWS 的初创公司和小企业，请查看 [AWS Marketplace 上的 Dify Premium](https://aws.amazon.com/marketplace/pp/prodview-t22mebxzwjhu6)，一键部署到您自己的 AWS VPC。这是一个价格实惠的 AMI 产品，可选择创建带有自定义徽标和品牌的应用程序。


## 保持领先

在 GitHub 上为 Dify 加星标，即时获得新版本通知。

![star-us](https://github.com/langgenius/dify/assets/13230914/b823edc1-6388-4e25-ad45-2f6b187adbb4)



## 快速开始
> 在安装 Dify 之前，请确保您的机器满足以下最低系统要求：
> 
>- CPU >= 2 核
>- RAM >= 4GB

</br>

启动 Dify 服务器最简单的方法是运行我们的 [docker-compose.yml](docker/docker-compose.yaml) 文件。在运行安装命令之前，请确保您的机器上已安装 [Docker](https://docs.docker.com/get-docker/) 和 [Docker Compose](https://docs.docker.com/compose/install/)：

```bash
cd docker
docker compose up -d
```

运行后，您可以在浏览器中访问 [http://localhost/install](http://localhost/install) 上的 Dify 仪表板并开始初始化过程。

> 如果您想为 Dify 做贡献或进行额外开发，请参考我们的[从源码部署指南](https://docs.dify.ai/getting-started/install-self-hosted/local-source-code)

## 下一步

如果您需要自定义配置，请参考我们 [docker-compose.yml](docker/docker-compose.yaml) 文件中的注释并手动设置环境配置。进行更改后，请再次运行 `docker-compose up -d`。您可以在[此处](https://docs.dify.ai/getting-started/install-self-hosted/environments)查看环境变量的完整列表。

如果您想配置高可用设置，有社区贡献的 [Helm Charts](https://helm.sh/) 允许在 Kubernetes 上部署 Dify。

- [Helm Chart by @LeoQuote](https://github.com/douban/charts/tree/master/charts/dify)
- [Helm Chart by @BorisPolonsky](https://github.com/BorisPolonsky/dify-helm)


## 贡献

对于想要贡献代码的人，请查看我们的[贡献指南](https://github.com/langgenius/dify/blob/main/CONTRIBUTING.md)。
同时，请考虑通过在社交媒体、活动和会议上分享来支持 Dify。


> 我们正在寻找贡献者帮助将 Dify 翻译成中文或英文以外的语言。如果您有兴趣帮助，请查看 [i18n README](https://github.com/langgenius/dify/blob/main/web/i18n/README.md) 了解更多信息，并在我们的 [Discord 社区服务器](https://discord.gg/8Tpq4AcN9c) 的 `global-users` 频道中给我们留言。

**Contributors**

<a href="https://github.com/langgenius/dify/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langgenius/dify" />
</a>

## 社区与联系

* [Github Discussion](https://github.com/langgenius/dify/discussions)。最适合：分享反馈和提问。
* [GitHub Issues](https://github.com/langgenius/dify/issues)。最适合：使用 Dify.AI 时遇到的错误和功能提议。请参阅我们的[贡献指南](https://github.com/langgenius/dify/blob/main/CONTRIBUTING.md)。
* [Email](mailto:support@dify.ai?subject=[GitHub]Questions%20About%20Dify)。最适合：关于使用 Dify.AI 的问题。
* [Discord](https://discord.gg/FngNHpbcY7)。最适合：分享您的应用程序并与社区交流。
* [Twitter](https://twitter.com/dify_ai)。最适合：分享您的应用程序并与社区交流。

或者，直接与团队成员安排会议：

<table>
  <tr>
    <th>联系人</th>
    <th>目的</th>
  </tr>
  <tr>
    <td><a href='https://cal.com/guchenhe/15min' target='_blank'><img class="schedule-button" src='https://github.com/langgenius/dify/assets/13230914/9ebcd111-1205-4d71-83d5-948d70b809f5' alt='Git-Hub-README-Button-3x' style="width: 180px; height: auto; object-fit: contain;"/></a></td>
    <td>商务咨询和产品反馈</td>
  </tr>
  <tr>
    <td><a href='https://cal.com/pinkbanana' target='_blank'><img class="schedule-button" src='https://github.com/langgenius/dify/assets/13230914/d1edd00a-d7e4-4513-be6c-e57038e143fd' alt='Git-Hub-README-Button-2x' style="width: 180px; height: auto; object-fit: contain;"/></a></td>
    <td>贡献、问题和功能请求</td>
  </tr>
</table>

## 星标历史

[![Star History Chart](https://api.star-history.com/svg?repos=langgenius/dify&type=Date)](https://star-history.com/#langgenius/dify&Date)


## 安全披露

为了保护您的隐私，请避免在 GitHub 上发布安全问题。相反，请将您的问题发送至 security@dify.ai，我们将为您提供更详细的答复。

## 许可证

本仓库在 [Dify 开源许可证](LICENSE) 下可用，本质上是 Apache 2.0 加上一些额外限制。
