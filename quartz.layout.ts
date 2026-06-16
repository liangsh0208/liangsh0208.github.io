import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [
    Component.ConditionalRender({
      component: Component.Graph({
        localGraph: undefined,
        globalGraph: {
          drag: true,
          zoom: true,
          depth: -1,
          scale: 0.9,
          repelForce: 0.2,
          centerForce: 0.15,
          linkDistance: 24,
          fontSize: 0.55,
          opacityScale: 1.2,
          showTags: true,
          removeTags: [],
        },
      }),
      condition: (page) => page.fileData.slug === "index",
    }),
    Component.ConditionalRender({
      component: Component.RecentNotes({
        title: "最近更新",
        limit: 8,
        linkToMore: false,
        showTags: true,
        filter: (f) => f.slug !== "index",
      }),
      condition: (page) => page.fileData.slug === "index",
    }),
  ],
  footer: Component.Footer({
    links: {
      GitHub: "https://github.com/liangsh0208",
    },
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ConditionalRender({
      component: Component.Breadcrumbs(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ConditionalRender({
      component: Component.ArticleTitle(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ConditionalRender({
      component: Component.ContentMeta(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ConditionalRender({
      component: Component.TagList(),
      condition: (page) => page.fileData.slug !== "index",
    }),
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
      ],
    }),
    Component.HomeLink(),
    Component.Explorer({
      folderDefaultState: "collapsed",
      folderClickBehavior: "link",
      title: "",
    }),
  ],
  right: [
    Component.DesktopOnly(Component.TableOfContents()),
    Component.DesktopOnly(Component.Graph({
      localGraph: {
        drag: true,
        zoom: true,
        depth: 1,
        scale: 1.1,
        repelForce: 0.5,
        centerForce: 0.3,
        linkDistance: 30,
        fontSize: 0.6,
        opacityScale: 1,
        showTags: false,
        removeTags: [],
      },
      globalGraph: undefined,
    })),
    Component.ConditionalRender({
      component: Component.TagCloud(),
      condition: (page) => page.fileData.slug === "index",
    }),
  ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
      ],
    }),
    Component.HomeLink(),
    Component.Explorer({
      folderDefaultState: "collapsed",
      folderClickBehavior: "link",
      title: "",
    }),
  ],
  right: [],
}
