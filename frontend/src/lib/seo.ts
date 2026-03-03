import { useEffect } from "react";

const SITE_NAME = "Tokemizer";
const DEFAULT_TITLE = "Tokemizer | Enterprise Prompt Token Optimizer & Compression";
const DEFAULT_DESCRIPTION =
  "Tokemizer is an enterprise-grade AI prompt token optimizer. Reduce LLM costs by 30-70% and improve latency while preserving semantic meaning, context, and intent. Secure, on-premise ready, and compatible with OpenAI, Anthropic, Google Gemini, and all major LLMs.";
const DEFAULT_IMAGE = "https://www.tokemizer.com/fav.png";
const DEFAULT_SITE_URL = "https://www.tokemizer.com";

export type StructuredData = Record<string, unknown>;

export interface SeoOptions {
  title?: string;
  description?: string;
  path?: string;
  image?: string;
  robots?: string;
  type?: "website" | "article";
  structuredData?: StructuredData | StructuredData[];
}

function upsertMeta(selector: string, attrs: Record<string, string>) {
  let tag = document.head.querySelector(selector) as HTMLMetaElement | null;
  if (!tag) {
    tag = document.createElement("meta");
    document.head.appendChild(tag);
  }

  Object.entries(attrs).forEach(([key, value]) => {
    tag?.setAttribute(key, value);
  });
}

function upsertLink(selector: string, attrs: Record<string, string>) {
  let tag = document.head.querySelector(selector) as HTMLLinkElement | null;
  if (!tag) {
    tag = document.createElement("link");
    document.head.appendChild(tag);
  }

  Object.entries(attrs).forEach(([key, value]) => {
    tag?.setAttribute(key, value);
  });
}

export function buildCanonicalUrl(path = "") {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${DEFAULT_SITE_URL}${normalizedPath}`;
}

export function useSeo({
  title,
  description,
  path,
  image,
  robots = "index,follow",
  type = "website",
  structuredData,
}: SeoOptions) {
  useEffect(() => {
    const pageTitle = title ? `${title} | ${SITE_NAME}` : DEFAULT_TITLE;
    const pageDescription = description ?? DEFAULT_DESCRIPTION;
    const canonical = buildCanonicalUrl(path ?? window.location.pathname);
    const previewImage = image ?? DEFAULT_IMAGE;

    document.title = pageTitle;

    upsertMeta('meta[name="description"]', {
      name: "description",
      content: pageDescription,
    });
    upsertMeta('meta[name="robots"]', { name: "robots", content: robots });

    upsertMeta('meta[property="og:title"]', { property: "og:title", content: pageTitle });
    upsertMeta('meta[property="og:description"]', {
      property: "og:description",
      content: pageDescription,
    });
    upsertMeta('meta[property="og:type"]', { property: "og:type", content: type });
    upsertMeta('meta[property="og:url"]', { property: "og:url", content: canonical });
    upsertMeta('meta[property="og:site_name"]', { property: "og:site_name", content: SITE_NAME });
    upsertMeta('meta[property="og:image"]', { property: "og:image", content: previewImage });

    upsertMeta('meta[name="twitter:card"]', { name: "twitter:card", content: "summary_large_image" });
    upsertMeta('meta[name="twitter:title"]', { name: "twitter:title", content: pageTitle });
    upsertMeta('meta[name="twitter:description"]', {
      name: "twitter:description",
      content: pageDescription,
    });
    upsertMeta('meta[name="twitter:image"]', { name: "twitter:image", content: previewImage });

    upsertLink('link[rel="canonical"]', { rel: "canonical", href: canonical });

    document
      .querySelectorAll("script[data-seo-ld]")
      .forEach((scriptTag) => scriptTag.parentNode?.removeChild(scriptTag));

    const schemas = structuredData ? (Array.isArray(structuredData) ? structuredData : [structuredData]) : [];

    schemas.forEach((schema) => {
      const scriptTag = document.createElement("script");
      scriptTag.type = "application/ld+json";
      scriptTag.setAttribute("data-seo-ld", "true");
      scriptTag.text = JSON.stringify(schema);
      document.head.appendChild(scriptTag);
    });
  }, [description, image, path, robots, structuredData, title, type]);
}

export const seoDefaults = {
  siteName: SITE_NAME,
  siteUrl: DEFAULT_SITE_URL,
};
