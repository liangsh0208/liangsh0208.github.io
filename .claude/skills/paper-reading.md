---
name: paper-reading
description: |
  深度阅读和分析学术论文，生成结构化的论文解读笔记（Obsidian 格式）。
  触发场景：用户提供论文 URL（arXiv、alphaxiv、OpenReview、ACL Anthology 等）或本地 PDF 路径，
  要求阅读、分析、总结论文内容。关键词包括：阅读论文、论文分析、paper reading、解读论文、
  论文笔记、读一下这篇、分析这篇 paper。
---

# 论文深度阅读 Skill

## 目标

将一篇学术论文转化为结构清晰、重点突出的中文解读笔记（Obsidian 格式），兼顾全局理解和算法细节，并自动提取论文关键图片。

---

## Obsidian 文档格式规范

所有输出必须严格遵守以下 Obsidian 规范：

### Frontmatter 格式

```yaml
---
created: YYYY-MM-DD
---
```

- 使用 `created` 字段（不是 `date` 或 `title`）
- 论文元信息（paper URL、code URL、authors）放在 frontmatter 之后的正文区域

### 文档头部结构

```markdown
---
created: {当前日期 YYYY-MM-DD}
---

paper: {论文URL}
code: {代码URL，没有则不写此行}
authors: {作者列表 (机构)}
tags: {3-5个关键标签}

# {论文标题}
```

### 图片引用格式

- **格式**: `![](文件名.png)` — 只写文件名，不含任何路径
- **存放位置**: 所有图片统一下载到 `Appendix/` 目录
  - 完整路径: `/Users/danchen/Documents/0.笔记文档/liangsh0208.github.io/content/Appendix/`
- **文件名中有空格时**: 用 `%20` 编码，如 `![](Pasted%20image%2020260512101327.png)`
- Obsidian 会自动在 vault 中查找匹配的文件名，无需写路径

### 文件名格式

`{简短标题或缩写}--{英文副标题}.md`

### 输出路径

默认: `/Users/danchen/Documents/0.笔记文档/liangsh0208.github.io/content/2.笔记/论文阅读/`

如果用户指定子目录则遵从用户指定。

---

## 执行流程

### Phase 1: 获取论文内容

根据输入类型选择获取方式：

1. **arXiv/alphaxiv URL**:
   - 提取 paper_id（如 `2606.06021`）
   - 先尝试 `https://arxiv.org/html/{paper_id}` 获取 HTML 全文（含图片 URL）
   - 用 WebFetch 提取核心内容
   - 用 tavily_extract 补充细节（方法、实验、公式），**设置 `include_images: true`** 获取图片列表
   - 用 tavily_search 搜索相关讨论和解读

2. **本地 PDF**:
   - 使用 Read 工具，通过 `pages` 参数分批读取（每次最多 20 页）
   - 第一次读 pages="1-10"，根据总页数继续
   - **注意**: PDF 中的图片无法通过 Read 工具直接提取为独立文件
   - 处理方式见下方「Phase 3 图片处理」

3. **其他 URL (OpenReview, ACL 等)**:
   - 尝试 WebFetch 直接抓取
   - 失败则用 tavily_extract（设置 `include_images: true`）
   - 必要时搜索 arXiv 版本以获取 HTML 全文和图片

### Phase 2: 信息提取（必须覆盖）

从论文中提取以下核心信息：

- [ ] 标题、作者、机构、发表会议/期刊
- [ ] 一句话核心贡献
- [ ] 研究动机（解决什么问题、现有方法有什么不足）
- [ ] 方法/算法（公式、架构、关键设计选择）
- [ ] 理论分析（定理、证明思路，如果有）
- [ ] 实验设置（数据集、基线、评估指标）
- [ ] 核心实验结果（主表格数据）
- [ ] 消融实验关键结论
- [ ] 局限性与未来方向
- [ ] 代码链接（如果有）
- [ ] **关键图片 URL 列表**（从 HTML 或 tavily_extract 获取）

### Phase 3: 图片处理

#### 规则：必须插入的图片类型

| 优先级 | 图片类型 | 说明 |
|--------|---------|------|
| P0 必须 | 方法架构图 / 流程图 | 论文的核心方法可视化 |
| P0 必须 | 主实验结果图 | 训练曲线、性能对比等核心 figure |
| P1 推荐 | Pareto 图 / 总览图 | 通常是 Figure 1，论文的"卖点图" |
| P1 推荐 | 消融实验图 | 关键 ablation 的可视化 |
| P2 可选 | 示例 / case study | 如果有助于理解方法 |

#### 按来源的图片获取策略

**来源 A: arXiv HTML 版论文（最佳来源）**

1. 通过 `tavily_extract` 设置 `include_images: true` 获取图片 URL 列表
2. 或从 HTML 页面中解析，图片 URL 通常为:
   - `https://arxiv.org/html/{paper_id}v{version}/Figure/xxx.png`
   - `https://arxiv.org/html/{paper_id}v{version}/x{N}.png`
3. 用 curl 下载到 Appendix 目录:
   ```bash
   curl -sL "{image_url}" -o "/Users/danchen/Documents/0.笔记文档/liangsh0208.github.io/content/Appendix/{文件名}.png"
   ```
4. 文件命名规则: `{论文缩写}_fig{N}_{简短描述}.png`
   - 例: `OPRD_fig1_pareto.png`, `OPRD_fig2_method.png`

**来源 B: 本地 PDF 文件**

PDF 中的图片无法通过 Read 工具直接提取。处理策略：

1. **首选**: 查找该论文是否有 arXiv HTML 版，从 HTML 获取图片（参考来源 A）
2. **次选**: 通过 tavily_search 搜索论文标题，找到在线版本提取图片
3. **兜底**: 在文档中用文字标注图片位置，提示用户手动补充:
   ```markdown
   > **[Figure X: {图片描述}]** — 此图来自 PDF 第 {N} 页，请在 Obsidian 中手动截图粘贴。
   ```

**来源 C: 其他在线 URL**

1. 优先从页面直接提取图片（tavily_extract + include_images）
2. 下载流程同来源 A

#### 图片插入格式

```markdown
![](OPRD_fig1_pareto.png)

> **Figure 1**: {图片说明，解释图中展示了什么、关键结论是什么}
```

- `![]()` 中只写文件名
- 紧跟 `>` 引用块作为图注，包含 Figure 编号和中文说明
- 多图并排时用 Markdown 表格:
  ```markdown
  | 场景A | 场景B | 场景C |
  |-------|-------|-------|
  | ![](fig_a.png) | ![](fig_b.png) | ![](fig_c.png) |
  ```

### Phase 4: 生成笔记文档

输出 Markdown 文件，使用以下模板结构：

```markdown
---
created: {YYYY-MM-DD}
---

paper: {论文URL}
code: {代码URL}
authors: {作者列表 (机构)}
tags: {标签}

# {论文标题}

## 一句话总结
{一句中文概括核心贡献和关键结果}

![](论文缩写_fig1_总览图.png)
> **Figure 1**: {总览图说明}

---

## 1. 研究背景与动机
### 1.1 问题定义
### 1.2 现有方法的不足

## 2. 方法
### 2.1 核心思想

![](论文缩写_fig2_method.png)
> **Figure 2**: {方法架构图说明}

### 2.2 算法/模型详解（含公式）
### 2.3 关键设计选择（为什么这样做）

## 3. 理论分析（如适用）
### 定理陈述 + 证明直觉 + 实际意义

## 4. 实验结果
### 4.1 实验设置
### 4.2 主实验结果（表格）

![](论文缩写_fig3_训练曲线.png)
> **Figure 3**: {实验结果图说明}

### 4.3 消融实验

![](论文缩写_fig4_ablation.png)
> **Figure 4**: {消融实验图说明}

### 4.4 训练/效率分析

## 5. 局限性与未来方向

## 6. 个人思考
{与用户当前研究方向的关联、方法的优雅/局限之处、启发}

## 7. 关键引用
{BibTeX}
```

### Phase 5: 质量检查

生成后自查：
- [ ] 公式是否完整且有逐项解释
- [ ] 核心实验数据是否准确引用（不是编造）
- [ ] 方法描述是否能让没读过原文的人理解
- [ ] 是否有"个人思考"部分（不是简单复述）
- [ ] **关键图片是否已插入**（至少包含方法图 + 主实验图）
- [ ] **图片文件是否已下载到 Appendix 目录**
- [ ] **图片引用格式是否只有文件名**（无路径前缀）
- [ ] **图注是否清晰解释了图的内容和结论**
- [ ] frontmatter 是否使用 `created` 字段

---

## 写作原则

1. **公式必须解释**: 每个公式写出后，紧跟各符号含义的解释表格或逐项说明
2. **先直觉后形式**: 每个方法/定理先给一句话直觉，再展开细节
3. **对比驱动**: 多用"相比 XX 方法，本文做了什么不同"的表达
4. **数据说话**: 核心结论必须有实验数据支撑
5. **中文为主**: 正文用中文，专有名词保留英文，公式用 LaTeX
6. **图文结合**: 关键位置必须有图片，图片紧跟相关文字段落

---

## 迭代优化记录

### v1.0 (2026-06-09)
- 初始版本：支持 arXiv URL 和本地 PDF，结构化模板输出，公式详解要求

### v1.1 (2026-06-09)
- 新增 Obsidian 文档格式规范（frontmatter 用 `created`、图片只写文件名）
- 新增图片处理完整流程（按来源分策略：arXiv HTML / 本地 PDF / 其他 URL）
- 新增图片优先级规则（P0 必须插入方法图和主实验图）
- 新增 PDF 论文图片的兜底处理方案
- 图片统一存放 Appendix 目录，命名规则 `{论文缩写}_fig{N}_{描述}.png`

### 待优化项
- [ ] 支持多论文对比阅读模式
- [ ] 支持增量更新（已有笔记追加新内容）
- [ ] 支持自动关联已有笔记中的相关论文
- [ ] 支持生成论文关系图谱
- [ ] 针对不同领域（NLP、CV、RL）微调 prompt 模板
