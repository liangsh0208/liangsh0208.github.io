import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"

interface Topic {
  title: string
  description: string
  slug: string
  count?: number
}

const topics: Topic[] = [
  {
    title: "Megatron-LM",
    description: "NVIDIA 大模型训练框架深度解析：并行策略、训练循环、推理部署等",
    slug: "infra-summary-with-cc/Megatron-LM",
  },
  {
    title: "LLM 基础",
    description: "大模型核心算法与论文解读：Transformer、注意力机制等",
    slug: "LLMcore",
  },
  {
    title: "随笔笔记",
    description: "技术思考、学习心得与日常记录",
    slug: "笔记",
  },
]

const TopicCards: QuartzComponent = ({
  fileData,
  displayClass,
}: QuartzComponentProps) => {
  return (
    <div class={classNames(displayClass, "topic-cards")}>
      <h2 class="topic-section-title">主题分类</h2>
      <div class="topic-grid">
        {topics.map((topic) => (
          <a href={`./${topic.slug}`} class="topic-card internal">
            <div class="topic-card-inner">
              <h3 class="topic-card-title">{topic.title}</h3>
              <p class="topic-card-desc">{topic.description}</p>
            </div>
          </a>
        ))}
      </div>
    </div>
  )
}

TopicCards.css = `
.topic-section-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--darkgray);
  margin-bottom: 1rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid rgba(128,128,128,0.1);
}

.topic-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.topic-card {
  display: block;
  text-decoration: none;
  border: 1px solid rgba(128,128,128,0.1);
  border-radius: 8px;
  padding: 1.1rem 1.25rem;
  background: rgba(128,128,128,0.02);
}

.topic-card-title {
  margin: 0 0 0.35rem 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--darkgray);
  line-height: 1.3;
}

.topic-card-desc {
  margin: 0;
  font-size: 0.82rem;
  line-height: 1.55;
  color: var(--gray);
}

:root[saved-theme="dark"] .topic-card {
  border-color: rgba(255,255,255,0.06);
  background: rgba(255,255,255,0.015);
}
`

export default (() => TopicCards) satisfies QuartzComponentConstructor
