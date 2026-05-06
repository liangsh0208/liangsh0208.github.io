import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import { pathToRoot } from "../util/path"

const ProfileCard: QuartzComponent = ({
  fileData,
  displayClass,
}: QuartzComponentProps) => {
  const rootPath = pathToRoot(fileData.slug || "")

  return (
    <div class={classNames(displayClass, "profile-card")}>
      <div class="profile-avatar">
        <img src={`${rootPath}static/icon.png`} alt="avatar" />
      </div>
      <div class="profile-body">
        <h1 class="profile-name">大模型 Infra 笔记</h1>
        <p class="profile-desc">
          聚焦大模型训练与推理全栈技术：并行策略、分布式训练框架（Megatron-LM、Hugging Face）、
          高性能算子内核（DeepEP、DeepGEMM、FlashMLA）以及存储系统（3FS）的深度源码解析。
        </p>
        <div class="profile-links">
          <a href="https://github.com/liangsh0208" target="_blank" rel="noopener" class="profile-link">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
            GitHub
          </a>
        </div>
      </div>
    </div>
  )
}

ProfileCard.css = `
.profile-card {
  display: flex;
  align-items: flex-start;
  gap: 1.25rem;
  padding: 1.5rem;
  margin-bottom: 2rem;
  border: 1px solid rgba(128,128,128,0.1);
  border-radius: 10px;
  background: rgba(128,128,128,0.02);
}

.profile-avatar {
  flex-shrink: 0;
  width: 64px;
  height: 64px;
  border-radius: 50%;
  overflow: hidden;
  background: var(--lightgray);
}

.profile-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.profile-body {
  flex: 1;
  min-width: 0;
}

.profile-name {
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0 0 0.4rem 0;
  color: var(--dark);
  line-height: 1.3;
}

.profile-desc {
  font-size: 0.9rem;
  line-height: 1.65;
  color: var(--gray);
  margin: 0 0 0.75rem 0;
}

.profile-links {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.profile-link {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.82rem;
  color: var(--secondary);
  text-decoration: none;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  background: var(--highlight);
}

.profile-link:hover {
  text-decoration: underline;
}

@media (max-width: 600px) {
  .profile-card {
    flex-direction: column;
    align-items: center;
    text-align: center;
  }
  .profile-links {
    justify-content: center;
  }
}

:root[saved-theme="dark"] .profile-card {
  background: rgba(255,255,255,0.015);
  border-color: rgba(255,255,255,0.06);
}
`

export default (() => ProfileCard) satisfies QuartzComponentConstructor
