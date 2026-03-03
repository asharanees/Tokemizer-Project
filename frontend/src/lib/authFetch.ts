import { toApiUrl } from "./apiUrl";

export const getAuthHeaders = () => {
  const token = localStorage.getItem("token");
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
};

let refreshPromise: Promise<string | null> | null = null;
type AuthRequestInit = RequestInit & { _retry?: boolean };

const refreshAccessToken = async (): Promise<string | null> => {
  const refreshToken = localStorage.getItem("refresh_token");
  if (!refreshToken) {
    localStorage.removeItem("token");
    return null;
  }
  if (!refreshPromise) {
    refreshPromise = (async () => {
      try {
        const response = await fetch(toApiUrl("/api/auth/refresh"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
        if (!response.ok) {
          localStorage.removeItem("token");
          localStorage.removeItem("refresh_token");
          return null;
        }
        const data = await response.json();
        if (!data.access_token) {
          localStorage.removeItem("token");
          localStorage.removeItem("refresh_token");
          return null;
        }
        localStorage.setItem("token", data.access_token);
        if (data.refresh_token) {
          localStorage.setItem("refresh_token", data.refresh_token);
        }
        return data.access_token as string;
      } finally {
        refreshPromise = null;
      }
    })();
  }
  return refreshPromise;
};

export async function authFetch(
  input: RequestInfo | URL,
  init: AuthRequestInit = {},
): Promise<Response> {
  const hasRetried = Boolean(init._retry);
  const headers = new Headers(init.headers || {});
  const authHeaders = getAuthHeaders();
  Object.entries(authHeaders).forEach(([key, value]) => headers.set(key, value));

  const resolvedInput =
    typeof input === "string" && input.startsWith("/") ? toApiUrl(input) : input;

  const response = await fetch(resolvedInput, { ...init, headers });
  const isUnauthorized = response.status === 401;
  const url =
    typeof resolvedInput === "string"
      ? resolvedInput
      : resolvedInput instanceof URL
        ? resolvedInput.toString()
        : resolvedInput.url;

  if (!isUnauthorized || hasRetried || url.includes("/api/auth/refresh")) {
    return response;
  }

  const newToken = await refreshAccessToken();
  if (!newToken) {
    return response;
  }

  const retryHeaders = new Headers(headers);
  retryHeaders.set("Authorization", `Bearer ${newToken}`);

  return authFetch(input, { ...init, headers: retryHeaders, _retry: true });
}
