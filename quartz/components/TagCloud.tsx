import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import { pathToRoot, resolveRelative, FullSlug } from "../util/path"

const TagCloud: QuartzComponent = ({
  allFiles,
  fileData,
  displayClass,
}: QuartzComponentProps) => {
  const rootPath = pathToRoot(fileData.slug || "")

  const tagCounts = new Map<string, number>()
  for (const f of allFiles || []) {
    const tags = f.frontmatter?.tags
    if (Array.isArray(tags)) {
      for (const tag of tags) {
        if (typeof tag === "string") {
          tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1)
        }
      }
    }
  }

  const sortedTags = Array.from(tagCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 24)

  if (sortedTags.length === 0) return null

  const maxCount = sortedTags[0][1]
  const minCount = sortedTags[sortedTags.length - 1][1]

  const fontSize = (count: number) => {
    if (maxCount === minCount) return "0.85rem"
    const t = (count - minCount) / (maxCount - minCount)
    return `${0.78 + t * 0.42}rem`
  }

  return (
    <div class={classNames(displayClass, "tag-cloud")}>
      <h2 class="tag-cloud-title">FEATURED TAGS</h2>
      <ul class="tag-cloud-list">
        {sortedTags.map(([tag, count]) => (
          <li>
            <a
              href={resolveRelative(fileData.slug!, `tags/${encodeURIComponent(tag)}` as FullSlug)}
              class="tag-cloud-link"
              style={`font-size: ${fontSize(count)}`}
            >
              {tag}
              <span class="tag-cloud-count">{count}</span>
            </a>
          </li>
        ))}
      </ul>
    </div>
  )
}

TagCloud.css = `
.tag-cloud {
  margin-bottom: 2rem;
}

.tag-cloud-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--darkgray);
  margin-bottom: 1rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid rgba(128,128,128,0.1);
}

.tag-cloud-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.tag-cloud-link {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.2rem 0.55rem;
  border-radius: 16px;
  background: var(--lightgray);
  color: var(--secondary);
  font-weight: 500;
  text-decoration: none;
  border: 1px solid rgba(128,128,128,0.12);
  transition: background 0.15s ease, border-color 0.15s ease;
}

.tag-cloud-link:hover {
  background: var(--highlight);
  border-color: rgba(128,128,128,0.2);
  text-decoration: none;
}

.tag-cloud-count {
  font-size: 0.7em;
  color: var(--gray);
  font-weight: 400;
  opacity: 0.7;
}

:root[saved-theme="dark"] .tag-cloud-link {
  background: rgba(255,255,255,0.04);
  border-color: rgba(255,255,255,0.06);
}

:root[saved-theme="dark"] .tag-cloud-link:hover {
  background: rgba(88,166,255,0.12);
}
`

export default (() => TagCloud) satisfies QuartzComponentConstructor
