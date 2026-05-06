import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot } from "../util/path"

const HomeLink: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const rootPath = pathToRoot(fileData.slug || "")
  return (
    <a href={rootPath} class="home-nav-link">
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
      Home
    </a>
  )
}

HomeLink.css = `
.home-nav-link {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.875rem;
  font-weight: 500;
  color: #34343c;
  text-decoration: none;
  padding: 0.4rem 0.6rem;
  margin-bottom: 0.5rem;
  border-radius: 6px;
}
.home-nav-link:hover {
  color: var(--secondary);
  background: var(--highlight);
}
:root[saved-theme="dark"] .home-nav-link {
  color: #c9d1d9;
}
:root[saved-theme="dark"] .home-nav-link:hover {
  color: #58a6ff;
}
`

export default (() => HomeLink) satisfies QuartzComponentConstructor
