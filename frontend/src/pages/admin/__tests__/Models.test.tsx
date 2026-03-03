import React from "react";
import { render, screen } from "@testing-library/react";
import { TooltipProvider } from "@radix-ui/react-tooltip";
import { vi } from "vitest";

import Models, {
    formatExpectedFiles,
    normalizeExpectedFiles,
    parseMinSizeBytes,
} from "../Models";

import * as rq from "@tanstack/react-query";

vi.mock("@/components/layout/Layout", () => ({
    Layout: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock("@/lib/authFetch", () => ({
    authFetch: vi.fn(),
}));

vi.mock("@/hooks/use-toast", () => ({
    useToast: () => ({ toast: vi.fn() }),
}));

const mockInvalidateQueries = vi.fn();

vi.mock("@tanstack/react-query", async () => {
    const actual = await vi.importActual<typeof import("@tanstack/react-query")>(
        "@tanstack/react-query"
    );
    return {
        ...actual,
        useQuery: vi.fn(),
        useMutation: vi.fn(),
        useQueryClient: vi.fn(() => ({ invalidateQueries: mockInvalidateQueries })),
    };
});

type ModelRefreshState = "idle" | "running" | "completed" | "failed";

const baseModelsResponse = {
    models: [
        {
            model_type: "semantic_guard",
            model_name: "example/model",
            component: "Guard",
            library_type: "transformers",
            usage: "scoring",
            min_size_bytes: 0,
            expected_files: ["config.json"],
            revision: "main",
            allow_patterns: [],
            size_bytes: 1024,
            size_formatted: "1 KB",
            download_date: null,
            cached_ok: false,
            cached_reason: "download_failed",
            cached_error_detail: "Failed to download after retries",
            loaded_ok: false,
            loaded_reason: "load_failed",
            intended_usage_ready: false,
            intended_usage_reason: "missing_required_model",
            intended_features: ["semantic_guard"],
            required_mode_gates: [],
            required_profile_gates: [],
            hard_required: true,
            last_refresh: "2026-02-13T00:00:00Z",
            path: null,
        },
    ],
    total_size_bytes: 1024,
    total_size_formatted: "1 KB",
    warnings: [],
};

const setupQueryMocks = (refreshState: ModelRefreshState) => {
    vi.spyOn(rq, "useQuery").mockImplementation((options: any) => {
        const key = options?.queryKey?.[0];
        if (key === "admin-models") {
            return {
                data: baseModelsResponse,
                isLoading: false,
                error: null,
            } as any;
        }
        if (key === "admin-model-refresh") {
            return {
                data: {
                    state: refreshState,
                    mode: "download_missing",
                    target_models: ["semantic_guard"],
                    warnings: [],
                },
                isLoading: false,
                error: null,
            } as any;
        }
        if (key === "admin-models-protected") {
            return {
                data: { protected_model_types: ["semantic_guard"] },
                isLoading: false,
                error: null,
            } as any;
        }
        return { data: undefined, isLoading: false, error: null } as any;
    });
};

const setupMutationMocks = () => {
    vi.spyOn(rq, "useMutation").mockImplementation(() => {
        return {
            mutate: vi.fn(),
            isPending: false,
        } as any;
    });
};

const renderWithProviders = (component: React.ReactElement) =>
    render(<TooltipProvider>{component}</TooltipProvider>);

describe("Models page", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        setupMutationMocks();
    });

    it("normalizes expected files from commas/newlines", () => {
        expect(normalizeExpectedFiles("a.json, b.json\nc.json")).toEqual([
            "a.json",
            "b.json",
            "c.json",
        ]);
        expect(normalizeExpectedFiles(undefined)).toEqual([]);
    });

    it("formats expected files as newline-separated text", () => {
        expect(formatExpectedFiles(["a.json", "b.json"])).toBe("a.json\nb.json");
        expect(formatExpectedFiles([])).toBe("");
    });

    it("parses min size bytes safely", () => {
        expect(parseMinSizeBytes("1024")).toBe(1024);
        expect(parseMinSizeBytes("abc")).toBeUndefined();
        expect(parseMinSizeBytes("")).toBeUndefined();
    });

    it("renders download-failed cached reason label", () => {
        setupQueryMocks("idle");
        renderWithProviders(<Models />);
        expect(
            screen.getByText("Download failed after retries")
        ).toBeInTheDocument();
    });

    it("shows running refresh banner and disables global refresh actions while running", () => {
        setupQueryMocks("running");
        renderWithProviders(<Models />);

        expect(
            screen.getByText(/Model refresh \(download_missing\) is running/)
        ).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: /download missing/i })
        ).toBeDisabled();
        expect(
            screen.getByRole("button", { name: /force redownload/i })
        ).toBeDisabled();
        expect(screen.getByRole("button", { name: /recovery run/i })).toBeDisabled();
        expect(screen.getByRole("button", { name: /add model/i })).toBeDisabled();
        expect(
            screen.getByRole("button", { name: /refresh semantic_guard/i })
        ).toBeDisabled();
        expect(screen.getByRole("button", { name: /edit semantic_guard/i })).toBeDisabled();
        expect(
            screen.getByRole("button", { name: /delete semantic_guard/i })
        ).toBeDisabled();
    });

    it("re-enables global refresh actions when refresh state is failed", () => {
        setupQueryMocks("failed");
        renderWithProviders(<Models />);

        expect(
            screen.getByRole("button", { name: /download missing/i })
        ).toBeEnabled();
        expect(
            screen.getByRole("button", { name: /force redownload/i })
        ).toBeEnabled();
        expect(screen.getByRole("button", { name: /recovery run/i })).toBeEnabled();
    });
});
