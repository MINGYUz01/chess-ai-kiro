# 下载配置

## 镜像源优先级
在终端下载包或内容时，应优先使用以下国内镜像源：

## NPM 镜像
- 淘宝 NPM 镜像: https://registry.npmmirror.com
```
npm config set registry https://registry.npmmirror.com
```

## Python 包镜像
- 清华大学镜像: https://pypi.tuna.tsinghua.edu.cn/simple
```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## Maven 镜像
- 阿里云 Maven 镜像: https://maven.aliyun.com/repository/public
```xml
<mirror>
  <id>aliyunmaven</id>
  <mirrorOf>central</mirrorOf>
  <name>阿里云公共仓库</name>
  <url>https://maven.aliyun.com/repository/public</url>
</mirror>
```

## Docker 镜像
- 阿里云 Docker 镜像: https://cr.console.aliyun.com/
```json
{
  "registry-mirrors": ["https://registry.cn-hangzhou.aliyuncs.com"]
}
```

## 其他包管理器
- 对于其他包管理器，应查找并使用相应的国内镜像源
- 在无法使用国内镜像源时，可以退而使用原始源

## 下载工具配置
- 使用支持断点续传的下载工具
- 配置适当的超时时间，以应对网络不稳定情况