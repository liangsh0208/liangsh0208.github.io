import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4 Configuration
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "我的技术博客",
    pageTitleSuffix: " - LLM 笔记",
    enableSPA: true,
    enablePopovers: true,
    analytics: {
      provider: "plausible",
    },
    locale: "zh-CN",
    baseUrl: "liangsh0208.github.io",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "created",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Inter",
        body: "Inter",
        code: "JetBrains Mono",
      },
      colors: {
        lightMode: {
          light: "#fafafa",
          lightgray: "#f3f0f8",
          gray: "#6b7280",
          darkgray: "#374151",
          dark: "#111827",
          secondary: "#6366f1",
          tertiary: "#6b7280",
          highlight: "rgba(99, 102, 241, 0.08)",
          textHighlight: "#fff17688",
        },
        darkMode: {
          light: "#0f0f12",
          lightgray: "#1c1c22",
          gray: "#9ca3af",
          darkgray: "#e5e7eb",
          dark: "#f9fafb",
          secondary: "#818cf8",
          tertiary: "#9ca3af",
          highlight: "rgba(129, 140, 248, 0.12)",
          textHighlight: "rgba(255, 248, 197, 0.2)",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description({
        descriptionLength: 800,
        maxDescriptionLength: 2000,
      }),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
      // Comment out CustomOgImages to speed up build time
      Plugin.CustomOgImages(),
    ],
  },
}

export default config
