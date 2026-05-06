import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import { pathToRoot } from "../util/path"

interface TreeNode {
  name: string
  slug: string
  children: TreeNode[]
}

function buildFolderTree(allFiles: any[]): TreeNode[] {
  const folderSet = new Set<string>()

  for (const f of allFiles) {
    const slug = f.slug
    if (!slug || slug === "index" || slug.startsWith("tags")) continue

    const parts = slug.split("/")
    if (parts.length < 2) continue

    for (let i = 1; i < parts.length; i++) {
      const prefix = parts.slice(0, i).join("/")
      if (prefix) folderSet.add(prefix)
    }
  }

  const roots: TreeNode[] = []

  for (const path of Array.from(folderSet).sort()) {
    const parts = path.split("/")
    let current = roots
    let currentPath = ""

    for (let i = 0; i < parts.length; i++) {
      const name = parts[i]
      currentPath = currentPath ? `${currentPath}/${name}` : name

      let node = current.find((n) => n.name === name)
      if (!node) {
        node = { name, slug: currentPath, children: [] }
        current.push(node)
      }
      current = node.children
    }
  }

  return roots
}

const FolderTree: QuartzComponent = ({
  allFiles,
  fileData,
  displayClass,
}: QuartzComponentProps) => {
  const tree = buildFolderTree(allFiles || [])
  const rootPath = pathToRoot(fileData.slug || "")

  if (tree.length === 0) return null

  const renderNode = (node: TreeNode, depth: number) => {
    const hasChildren = node.children.length > 0
    const isOpen = depth === 0

    return (
      <li class={`folder-tree-item depth-${depth}`}>
        <div class="folder-tree-row">
          {hasChildren ? (
            <button class="folder-tree-toggle" aria-expanded={isOpen}>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="12"
                height="12"
                viewBox="5 8 14 8"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                class="folder-icon"
              >
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>
          ) : (
            <span class="folder-tree-spacer" />
          )}
          <a href={`${rootPath}${node.slug}/`} class="folder-tree-link">
            {node.name}
          </a>
        </div>
        {hasChildren && (
          <ul class={`folder-tree-children ${isOpen ? "open" : "collapsed"}`}>
            {node.children.map((child) => renderNode(child, depth + 1))}
          </ul>
        )}
      </li>
    )
  }

  return (
    <div class={classNames(displayClass, "folder-tree")}>
      <h2 class="folder-tree-title">文档目录</h2>
      <ul class="folder-tree-root">
        {tree.map((node) => renderNode(node, 0))}
      </ul>
    </div>
  )
}

FolderTree.css = `
.folder-tree {
  margin-bottom: 2rem;
}
.folder-tree-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--darkgray);
  margin-bottom: 1rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid rgba(128,128,128,0.1);
}
.folder-tree-root,
.folder-tree-children {
  list-style: none;
  padding: 0;
  margin: 0;
}
.folder-tree-children {
  margin-left: 1.2rem;
  padding-left: 0.35rem;
  border-left: 1px solid #e0e0e0;
  display: none;
}
.folder-tree-children.open {
  display: block;
}
.folder-tree-children.collapsed {
  display: none;
}
.folder-tree-row {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.3rem 0;
}
.folder-tree-toggle {
  background: none;
  border: none;
  padding: 0;
  cursor: pointer;
  color: #8c959f;
  display: flex;
  align-items: center;
  flex-shrink: 0;
}
.folder-tree-toggle .folder-icon {
  transition: transform 0.2s ease;
}
.folder-tree-toggle[aria-expanded="true"] .folder-icon {
  transform: rotate(180deg);
}
.folder-tree-spacer {
  display: inline-block;
  width: 12px;
  flex-shrink: 0;
}
.folder-tree-link {
  font-size: 0.875rem;
  color: #34343c;
  font-weight: 500;
  text-decoration: none;
}
.folder-tree-link:hover {
  color: var(--secondary);
}
`

FolderTree.afterDOMLoaded = `
function setupFolderTree() {
  document.querySelectorAll('.folder-tree-toggle').forEach((btn) => {
    btn.addEventListener('click', function () {
      const expanded = this.getAttribute('aria-expanded') === 'true'
      this.setAttribute('aria-expanded', String(!expanded))
      const children = this.closest('.folder-tree-item').querySelector('.folder-tree-children')
      if (children) {
        children.classList.toggle('open')
        children.classList.toggle('collapsed')
      }
    })
  })
}
document.addEventListener("nav", setupFolderTree)
`

export default (() => FolderTree) satisfies QuartzComponentConstructor
