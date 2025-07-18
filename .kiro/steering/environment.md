# 计算机环境配置

## 操作系统
- **操作系统**: Windows 10
- **终端**: CMD (命令提示符)
- **Shell**: CMD

## 命令行兼容性
- 所有命令必须与 Windows CMD 兼容
- 不要使用 PowerShell 特有的命令（如 `Get-ChildItem`、`Get-Content` 等）
- 不要使用 Unix/Linux 特有的命令（如 `ls`、`cat`、`grep` 等）

## Windows CMD 命令替代参考
| Unix/Linux 命令 | PowerShell 命令 | Windows CMD 命令 |
|----------------|----------------|-----------------|
| ls             | Get-ChildItem   | dir             |
| cat            | Get-Content     | type            |
| grep           | Select-String   | findstr         |
| rm             | Remove-Item     | del             |
| cp             | Copy-Item       | copy            |
| mv             | Move-Item       | move            |
| mkdir          | New-Item -ItemType Directory | mkdir |
| touch          | New-Item        | echo.> filename |
| pwd            | Get-Location    | cd              |
| echo           | Write-Output    | echo            |
| find           | Get-ChildItem -Recurse | dir /s   |

## 路径格式
- 使用反斜杠 `\` 作为路径分隔符（而非正斜杠 `/`）
- 示例: `C:\Users\username\Documents`

## 环境变量
- 使用 `%VARIABLE%` 格式引用环境变量（而非 `$VARIABLE`）
- 示例: `%USERPROFILE%\Documents`

## 命令连接符
- 使用 `&` 连接多个命令（而非 `;` 或 `&&`）
- 示例: `mkdir temp & cd temp`

## 重定向
- 输出重定向: `command > file.txt`
- 追加输出: `command >> file.txt`
- 错误重定向: `command 2> error.txt`

## 批处理脚本
- 使用 `.bat` 或 `.cmd` 作为批处理脚本扩展名
- 批处理脚本中的变量使用 `%variable%` 格式

## 注意事项
- 每次提供命令时，确保命令适用于 Windows CMD
- 避免使用需要管理员权限的命令，除非明确指示
- 提供命令时，优先考虑 Windows 原生工具和命令