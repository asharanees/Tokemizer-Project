const normalizedApiBaseUrl = (import.meta.env.VITE_API_BASE_URL || "")
  .trim()
  .replace(/\/+$/, "");

export function toApiUrl(path: string): string {
  if (!path.startsWith("/")) {
    return path;
  }
  if (!normalizedApiBaseUrl) {
    return path;
  }
  return `${normalizedApiBaseUrl}${path}`;
}
