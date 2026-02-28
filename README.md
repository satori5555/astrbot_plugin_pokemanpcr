# astrbot_plugin_pokemanpcr

 `poke-man-pcr` Hoshino 项目迁移到 AstrBot 的一版完整插件。

## 已完成的迁移

- 去掉了 Hoshino 的 `Service`、`CQEvent`、`NoticeSession` 依赖
- 改为 AstrBot 的消息事件监听模型
- 去掉了 `DailyNumberLimiter` / `FreqLimiter` 依赖，改为插件内实现
- 去掉了 `chara` 依赖，改为直接使用 `_pcr_data.py` 构建角色别名索引
- 数据持久化改为 `data/plugin_data/astrbot_plugin_pokemanpcr/poke_man_pcr.db`
- 保留原始图片框资源，并增加“无头像资源时的占位卡面”兜底
- 支持文字“戳”、OneBot `Poke` 消息段、查看仓库、合成、交换、赠送、仓库对比、刷新卡池

## 安装

1. 将整个目录放到 `AstrBot/data/plugins/astrbot_plugin_pokemanpcr`
2. 安装依赖：AstrBot 会按 `requirements.txt` 自动安装，或手动安装 `Pillow`
3. 在 AstrBot WebUI 重载插件

## 头像资源

原始 Hoshino 插件依赖 `img/priconne/unit/icon_unit_*.png`。

- 如果你有这套资源，请在插件配置 `unit_image_dir` 中填写该目录路径
- 如果没有，插件会自动使用内置占位卡面，功能仍可运行，只是卡图会变成文字占位

## 指令

支持两种触发方式：

- 直接发送中文命令（兼容原项目）
- 或发送带 `/` 的同名命令

主要命令：

- `戳`
- `查看仓库 [@某人]`
- `仓库对比 @某人`
- `合成 卡1 卡2`
- `一键合成 稀有度1 稀有度2 [轮数]`
- `赠送 @某人 卡名`
- `交换 卡1 @某人 卡2`
- `确认交换`
- `刷新卡片`
- `/pcrcards`（帮助）

## 说明

这版代码已经完成框架层面的迁移，但仍建议你在目标 AstrBot 实例里实际重载一次，重点验证：

- 你所用适配器的 `At` 消息段解析
- 你所用适配器的 `Poke` 消息段是否按预期上报
- 你的运行环境里 `Arial` 不存在时的中文字体回退效果


## 头像目录说明

如果你的头像资源放在 `main.py` 同级的 `unit__` 文件夹（例如 `D:\AstrBot\data\plugins\astrbot_plugin_pokemanpcr\unit__`），现在无需额外配置：

- `unit_image_dir` 留空即可
- 插件会自动优先读取同级 `unit__`
- 若手动填写相对路径，也会按插件目录解析
