import { useEffect, useRef, useState, type ChangeEvent } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { authFetch } from "@/lib/authFetch";
import { Loader2, HardDrive, CheckCircle2, Edit2, AlertCircle, Trash2, Plus, RefreshCw, Upload } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface ModelInfo {
    model_type: string;
    model_name: string;
    component: string;
    library_type: string;
    usage: string;
    min_size_bytes?: number;
    expected_files?: string[];
    revision?: string | null;
    allow_patterns?: string[];
    size_bytes: number;
    size_formatted: string;
    download_date: string | null;
    cached_ok: boolean;
    cached_reason?: string | null;
    cached_error_detail?: string | null;
    loaded_ok?: boolean | null;
    loaded_reason?: string | null;
    intended_usage_ready?: boolean | null;
    intended_usage_reason?: string | null;
    intended_features?: string[];
    required_mode_gates?: string[];
    required_profile_gates?: string[];
    hard_required?: boolean;
    last_refresh?: string | null;
    path: string | null;
}

interface ModelListResponse {
    models: ModelInfo[];
    total_size_bytes: number;
    total_size_formatted: string;
    warnings?: string[];
}

interface ModelRefreshStatus {
    state: "idle" | "running" | "completed" | "failed";
    started_at?: string | null;
    finished_at?: string | null;
    error?: string | null;
    mode?: string | null;
    available_models?: string[];
    missing_models?: string[];
    target_models?: string[];
    warnings?: string[];
}

interface AirgapReadinessResponse {
    ready: boolean;
    missing_models: string[];
    invalid_models: string[];
    manifest_failures: Record<string, string>;
    checked_at: string;
}

interface UpdateModelParams {
    model_type: string;
    model_name: string;
    component?: string;
    library_type?: string;
    usage?: string;
    min_size_bytes?: number;
    expected_files?: string[];
    revision?: string;
    allow_patterns?: string[];
}

interface CreateModelParams {
    model_type: string;
    model_name: string;
    component?: string;
    library_type?: string;
    usage?: string;
    min_size_bytes?: number;
    expected_files?: string[];
    revision?: string;
    allow_patterns?: string[];
}

interface DeleteModelParams {
    modelType: string;
    overrideCore?: boolean;
}

export const normalizeExpectedFiles = (input?: string): string[] => {
    if (!input) {
        return [];
    }

    return input
        .split(/[\n,]+/)
        .map((item) => item.trim())
        .filter(Boolean);
};

export const formatExpectedFiles = (files?: string[]): string => {
    if (!files || files.length === 0) {
        return "";
    }
    return files.join("\n");
};

export const parseMinSizeBytes = (value: string) => {
    const normalizedValue = value.trim();
    if (!normalizedValue) {
        return undefined;
    }
    const parsed = Number(normalizedValue);
    if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed < 0) {
        return undefined;
    }
    return parsed;
};

export const PROTECTED_MODEL_TYPES = new Set([
    "semantic_guard",
    "semantic_rank",
    "entropy",
    "entropy_fast",
    "token_classifier",
    "coreference",
]);

interface ProtectedModelTypesResponse {
    protected_model_types: string[];
}

function EditModelDialog({ model, open, onOpenChange }: { model: ModelInfo, open: boolean, onOpenChange: (open: boolean) => void }) {
    const [modelName, setModelName] = useState(model.model_name);
    const [component, setComponent] = useState(model.component || "");
    const [libraryType, setLibraryType] = useState(model.library_type || "");
    const [usage, setUsage] = useState(model.usage || "");
    const [minSizeBytes, setMinSizeBytes] = useState(model.min_size_bytes?.toString() || "");
    const [expectedFiles, setExpectedFiles] = useState(formatExpectedFiles(model.expected_files));
    const [revision, setRevision] = useState(model.revision || "");
    const [allowPatterns, setAllowPatterns] = useState(formatExpectedFiles(model.allow_patterns));
    const { toast } = useToast();
    const queryClient = useQueryClient();

    useEffect(() => {
        setModelName(model.model_name);
        setComponent(model.component || "");
        setLibraryType(model.library_type || "");
        setUsage(model.usage || "");
        setMinSizeBytes(model.min_size_bytes?.toString() || "");
        setExpectedFiles(formatExpectedFiles(model.expected_files));
        setRevision(model.revision || "");
        setAllowPatterns(formatExpectedFiles(model.allow_patterns));
    }, [model]);

    const mutation = useMutation({
        mutationFn: async (params: UpdateModelParams) => {
            const res = await authFetch(`/api/admin/models/${params.model_type}`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model_name: params.model_name,
                    component: params.component,
                    library_type: params.library_type,
                    usage: params.usage,
                    min_size_bytes: params.min_size_bytes,
                    expected_files: params.expected_files,
                    revision: params.revision,
                    allow_patterns: params.allow_patterns,
                }),
            });
            if (!res.ok) {
                const error = await res.json();
                throw new Error(error.detail || "Failed to update model");
            }
            return res.json();
        },
        onSuccess: () => {
            toast({
                title: "Model updated",
                description: "Model configuration saved. A background refresh has been queued for this model; readiness will update when refresh completes.",
            });
            queryClient.invalidateQueries({ queryKey: ["admin-model-refresh"] });
            queryClient.invalidateQueries({ queryKey: ["admin-models"] });
            onOpenChange(false);
        },
        onError: (error: Error) => {
            toast({
                variant: "destructive",
                title: "Error",
                description: error.message,
            });
        },
    });

    const handleSave = () => {
        if (!modelName.trim()) {
            toast({
                variant: "destructive",
                title: "Validation Error",
                description: "Model name cannot be empty",
            });
            return;
        }
        const parsedExpected = normalizeExpectedFiles(expectedFiles);
        const parsedAllowPatterns = normalizeExpectedFiles(allowPatterns);
        const minSizeInput = minSizeBytes.trim();
        const minSizeValue = parseMinSizeBytes(minSizeBytes);
        if (minSizeInput && minSizeValue === undefined) {
            toast({
                variant: "destructive",
                title: "Validation Error",
                description: "Min size must be a non-negative whole number",
            });
            return;
        }
        mutation.mutate({
            model_type: model.model_type,
            model_name: modelName,
            component,
            library_type: libraryType,
            usage,
            min_size_bytes: minSizeValue,
            expected_files: parsedExpected,
            revision: revision.trim() || undefined,
            allow_patterns: parsedAllowPatterns,
        });
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent>
                <DialogHeader>
                    <DialogTitle>Edit Model Configuration</DialogTitle>
                    <DialogDescription>
                        Change the HuggingFace model identifier. This will trigger a new download if the model is not already cached.
                    </DialogDescription>
                </DialogHeader>
                    <div className="grid gap-4 py-4">
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="model-name" className="text-right">
                            Model Name
                        </Label>
                        <Input
                            id="model-name"
                            value={modelName}
                            onChange={(e) => setModelName(e.target.value)}
                            className="col-span-3"
                            placeholder="e.g. BAAI/bge-small-en-v1.5"
                        />
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="component" className="text-right">
                            Component
                        </Label>
                        <Input
                            id="component"
                            value={component}
                            onChange={(e) => setComponent(e.target.value)}
                            className="col-span-3"
                            placeholder="e.g. Semantic Guard"
                        />
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="library-type" className="text-right">
                            Library / Type
                        </Label>
                        <Input
                            id="library-type"
                            value={libraryType}
                            onChange={(e) => setLibraryType(e.target.value)}
                            className="col-span-3"
                            placeholder="e.g. sentence-transformers"
                        />
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="usage" className="text-right">
                            Usage
                        </Label>
                        <Input
                            id="usage"
                            value={usage}
                            onChange={(e) => setUsage(e.target.value)}
                            className="col-span-3"
                            placeholder="e.g. Semantic similarity scoring"
                        />
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="min-size-bytes" className="text-right">
                            Min Size (bytes)
                        </Label>
                        <Input
                            id="min-size-bytes"
                            type="number"
                            step="1"
                            value={minSizeBytes}
                            onChange={(e) => setMinSizeBytes(e.target.value)}
                            className="col-span-3"
                            placeholder="e.g. 104857600"
                        />
                    </div>
                    <div className="grid grid-cols-4 items-start gap-4">
                        <Label htmlFor="expected-files" className="text-right">
                            Expected Files
                        </Label>
                        <div className="col-span-3 space-y-2">
                            <Textarea
                                id="expected-files"
                                value={expectedFiles}
                                onChange={(e) => setExpectedFiles(e.target.value)}
                                className="h-24"
                                placeholder='e.g. model.safetensors\nconfig.json'
                            />
                            <p className="text-xs text-muted-foreground">
                                Enter newline or comma separated filenames (e.g. ["model.safetensors","config.json"]).
                            </p>
                        </div>
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="revision" className="text-right">
                            Revision
                        </Label>
                        <Input
                            id="revision"
                            value={revision}
                            onChange={(e) => setRevision(e.target.value)}
                            className="col-span-3"
                            placeholder="e.g. main or commit hash"
                        />
                    </div>
                    <div className="grid grid-cols-4 items-start gap-4">
                        <Label htmlFor="allow-patterns" className="text-right">
                            Allow Patterns
                        </Label>
                        <div className="col-span-3 space-y-2">
                            <Textarea
                                id="allow-patterns"
                                value={allowPatterns}
                                onChange={(e) => setAllowPatterns(e.target.value)}
                                className="h-24"
                                placeholder="e.g. *.json\n*.safetensors"
                            />
                            <p className="text-xs text-muted-foreground">
                                Optional override for download patterns (newline or comma separated).
                            </p>
                        </div>
                    </div>
                </div>
                <DialogFooter>
                    <Button variant="outline" onClick={() => onOpenChange(false)} tooltip="Cancel editing this model">Cancel</Button>
                    <Button onClick={handleSave} disabled={mutation.isPending} tooltip="Save changes to this model">
                        {mutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                        Save Changes
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}

function AddModelDialog({ open, onOpenChange }: { open: boolean, onOpenChange: (open: boolean) => void }) {
    const [modelType, setModelType] = useState("");
    const [modelName, setModelName] = useState("");
    const [component, setComponent] = useState("");
    const [libraryType, setLibraryType] = useState("");
    const [usage, setUsage] = useState("");
    const [minSizeBytes, setMinSizeBytes] = useState("");
    const [expectedFiles, setExpectedFiles] = useState("");
    const [revision, setRevision] = useState("");
    const [allowPatterns, setAllowPatterns] = useState("");
    const { toast } = useToast();
    const queryClient = useQueryClient();

    const mutation = useMutation({
        mutationFn: async (params: CreateModelParams) => {
            const res = await authFetch("/api/admin/models", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(params),
            });
            if (!res.ok) {
                const error = await res.json();
                throw new Error(error.detail || "Failed to create model");
            }
            return res.json();
        },
        onSuccess: () => {
            toast({
                title: "Model created",
                description: "Model configuration saved. A background refresh has been queued for the new model.",
            });
            queryClient.invalidateQueries({ queryKey: ["admin-models"] });
            onOpenChange(false);
            setModelType("");
            setModelName("");
            setComponent("");
            setLibraryType("");
            setUsage("");
            setMinSizeBytes("");
            setExpectedFiles("");
            setRevision("");
            setAllowPatterns("");
        },
        onError: (error: Error) => {
            toast({
                variant: "destructive",
                title: "Error",
                description: error.message,
            });
        },
    });

    const handleSave = () => {
        if (!modelType.trim() || !modelName.trim()) {
            toast({
                variant: "destructive",
                title: "Validation Error",
                description: "Model type and name cannot be empty",
            });
            return;
        }
        const parsedExpected = normalizeExpectedFiles(expectedFiles);
        const parsedAllowPatterns = normalizeExpectedFiles(allowPatterns);
        const minSizeInput = minSizeBytes.trim();
        const minSizeValue = parseMinSizeBytes(minSizeBytes);
        if (minSizeInput && minSizeValue === undefined) {
            toast({
                variant: "destructive",
                title: "Validation Error",
                description: "Min size must be a non-negative whole number",
            });
            return;
        }
        mutation.mutate({
            model_type: modelType.trim(),
            model_name: modelName.trim(),
            component: component.trim(),
            library_type: libraryType.trim(),
            usage: usage.trim(),
            min_size_bytes: minSizeValue,
            expected_files: parsedExpected,
            revision: revision.trim() || undefined,
            allow_patterns: parsedAllowPatterns,
        });
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent>
                <DialogHeader>
                    <DialogTitle>Add New Model</DialogTitle>
                    <DialogDescription>
                        Register a new model type and its HuggingFace identifier.
                    </DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="new-model-type" className="text-right">
                            Type (Key)
                        </Label>
                        <Input
                            id="new-model-type"
                            value={modelType}
                            onChange={(e) => setModelType(e.target.value)}
                            className="col-span-3"
                            placeholder="e.g. summarizer_model"
                        />
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="new-model-name" className="text-right">
                            Model Name
                        </Label>
                        <Input
                            id="new-model-name"
                            value={modelName}
                            onChange={(e) => setModelName(e.target.value)}
                            className="col-span-3"
                            placeholder="e.g. facebook/bart-large-cnn"
                        />
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="new-component" className="text-right">
                            Component
                        </Label>
                        <Input
                            id="new-component"
                            value={component}
                            onChange={(e) => setComponent(e.target.value)}
                            className="col-span-3"
                            placeholder="e.g. Summarizer"
                        />
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="new-library-type" className="text-right">
                            Library / Type
                        </Label>
                        <Input
                            id="new-library-type"
                            value={libraryType}
                            onChange={(e) => setLibraryType(e.target.value)}
                            className="col-span-3"
                            placeholder="e.g. transformers"
                        />
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="new-usage" className="text-right">
                            Usage
                        </Label>
                        <Input
                            id="new-usage"
                            value={usage}
                            onChange={(e) => setUsage(e.target.value)}
                            className="col-span-3"
                            placeholder="e.g. Summarization"
                        />
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="new-min-size-bytes" className="text-right">
                            Min Size (bytes)
                        </Label>
                        <Input
                            id="new-min-size-bytes"
                            type="number"
                            step="1"
                            value={minSizeBytes}
                            onChange={(e) => setMinSizeBytes(e.target.value)}
                            className="col-span-3"
                            placeholder="e.g. 104857600"
                        />
                    </div>
                    <div className="grid grid-cols-4 items-start gap-4">
                        <Label htmlFor="new-expected-files" className="text-right">
                            Expected Files
                        </Label>
                        <div className="col-span-3 space-y-2">
                            <Textarea
                                id="new-expected-files"
                                value={expectedFiles}
                                onChange={(e) => setExpectedFiles(e.target.value)}
                                className="h-24"
                                placeholder='e.g. model.safetensors\nconfig.json'
                            />
                            <p className="text-xs text-muted-foreground">
                                Enter newline or comma separated filenames (e.g. ["model.safetensors","config.json"]).
                            </p>
                        </div>
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="new-revision" className="text-right">
                            Revision
                        </Label>
                        <Input
                            id="new-revision"
                            value={revision}
                            onChange={(e) => setRevision(e.target.value)}
                            className="col-span-3"
                            placeholder="e.g. main or commit hash"
                        />
                    </div>
                    <div className="grid grid-cols-4 items-start gap-4">
                        <Label htmlFor="new-allow-patterns" className="text-right">
                            Allow Patterns
                        </Label>
                        <div className="col-span-3 space-y-2">
                            <Textarea
                                id="new-allow-patterns"
                                value={allowPatterns}
                                onChange={(e) => setAllowPatterns(e.target.value)}
                                className="h-24"
                                placeholder="e.g. *.json\n*.safetensors"
                            />
                            <p className="text-xs text-muted-foreground">
                                Optional override for download patterns (newline or comma separated).
                            </p>
                        </div>
                    </div>
                </div>
                <DialogFooter>
                    <Button variant="outline" onClick={() => onOpenChange(false)} tooltip="Cancel adding new model">Cancel</Button>
                    <Button onClick={handleSave} disabled={mutation.isPending} tooltip="Add this new model to the system">
                        {mutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                        Add Model
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}

export default function Models() {
    const { toast } = useToast();
    const queryClient = useQueryClient();
    const { data, isLoading, error } = useQuery<ModelListResponse>({
        queryKey: ["admin-models"],
        queryFn: async () => {
            const res = await authFetch("/api/admin/models");
            if (!res.ok) throw new Error("Failed to fetch models");
            return res.json();
        },
    });

    const { data: refreshStatus, error: refreshError } = useQuery<ModelRefreshStatus>({
        queryKey: ["admin-model-refresh"],
        queryFn: async () => {
            const res = await authFetch("/api/admin/models/refresh");
            if (!res.ok) throw new Error("Failed to fetch refresh status");
            return res.json();
        },
        refetchInterval: (query) =>
            query.state.data?.state === "running" ? 2000 : false,
    });
    const { data: protectedModelTypes } = useQuery<ProtectedModelTypesResponse>({
        queryKey: ["admin-models-protected"],
        queryFn: async () => {
            const res = await authFetch("/api/admin/models/protected");
            if (!res.ok) throw new Error("Failed to fetch protected model types");
            return res.json();
        },
        staleTime: 300000,
    });

    const refreshStateRef = useRef<ModelRefreshStatus["state"]>("idle");
    const refreshErrorRef = useRef<string | null>(null);
    const uploadInputRef = useRef<HTMLInputElement | null>(null);
    const [uploadTargetModel, setUploadTargetModel] = useState<string | null>(null);

    const [editingModel, setEditingModel] = useState<ModelInfo | null>(null);
    const [isAddOpen, setIsAddOpen] = useState(false);
    const [deleteModelType, setDeleteModelType] = useState<string | null>(null);
    const cachedReasonLabels: Record<string, string> = {
        cached_ok: "Cached OK",
        cache_missing: "Cache missing",
        cache_invalid: "Cache invalid",
        auth_failed: "Auth/access failed",
        auth_blocked: "Auth/access blocked",
        download_failed: "Download failed after retries",
        size_too_small: "Below minimum size",
        missing_files: "Missing files",
        manifest_missing: "Manifest missing",
        manifest_invalid: "Manifest invalid",
        manifest_version_mismatch: "Manifest version mismatch",
        manifest_file_missing: "Manifest file missing",
        manifest_size_mismatch: "Manifest size mismatch",
        manifest_hash_mismatch: "Manifest hash mismatch",
        revision_mismatch: "Revision mismatch",
        snapshot_missing: "Snapshot missing",
        validation_error: "Validation error",
        locked: "Refresh locked",
    };
    const loadedReasonLabels: Record<string, string> = {
        loaded_ok: "Loaded OK",
        not_warmed: "Warm-up not run",
        not_refreshed: "Refresh not run yet",
        load_failed: "Load failed",
    };

    const getIntendedUsageBadgeLabel = (model: ModelInfo): string => {
        const isSpacy = model.model_type === "spacy";
        if (model.intended_usage_reason === "alternate_backend_loaded") {
            return "Ready (alternate backend)";
        }
        if (model.intended_usage_ready === true) {
            return isSpacy
                ? "Ready (Semantic deduplication and linguistic passes)"
                : "Ready for intended usage";
        }
        if (isSpacy) {
            return "Not Ready (Semantic deduplication and linguistic passes)";
        }
        return model.hard_required ? "Not Ready (Required)" : "Not Ready (Optional)";
    };

    const getLoadedStatusBadgeLabel = (model: ModelInfo): string => {
        if (model.loaded_ok === true) {
            return "Loaded OK";
        }
        if (model.loaded_reason === "load_failed") {
            return "Failed";
        }
        return "Not Warmed";
    };

    const deleteMutation = useMutation({
        mutationFn: async ({ modelType, overrideCore }: DeleteModelParams) => {
            const query = overrideCore ? "?override_core_models=true" : "";
            const res = await authFetch(`/api/admin/models/${modelType}${query}`, {
                method: "DELETE",
            });
            if (!res.ok) {
                const error = await res.json();
                throw new Error(error.detail || "Failed to delete model");
            }
            return res.json();
        },
        onSuccess: () => {
            toast({
                title: "Model deleted",
                description: "Model configuration removed.",
            });
            queryClient.invalidateQueries({ queryKey: ["admin-models"] });
        },
        onError: (error: Error) => {
            toast({
                variant: "destructive",
                title: "Error",
                description: error.message,
            });
        },
    });
    const refreshMutation = useMutation({
        mutationFn: async (mode: string) => {
            const res = await authFetch(`/api/admin/models/refresh?mode=${mode}`, {
                method: "POST",
            });
            if (!res.ok) {
                const error = await res.json();
                throw new Error(error.detail || "Failed to refresh models");
            }
            return res.json();
        },
        onSuccess: (_status, mode) => {
            toast({
                title: "Model refresh started",
                description: `Refresh mode: ${mode.replace(/_/g, " ")}.`,
            });
            queryClient.invalidateQueries({ queryKey: ["admin-model-refresh"] });
            queryClient.invalidateQueries({ queryKey: ["admin-models"] });
        },
        onError: (error: Error) => {
            toast({
                variant: "destructive",
                title: "Error",
                description: error.message,
            });
        },
    });
    const refreshModelMutation = useMutation({
        mutationFn: async ({ modelType, mode }: { modelType: string; mode: string }) => {
            const res = await authFetch(
                `/api/admin/models/${modelType}/refresh?mode=${mode}`,
                {
                    method: "POST",
                }
            );
            if (!res.ok) {
                const error = await res.json();
                throw new Error(error.detail || "Failed to refresh model");
            }
            return res.json();
        },
        onSuccess: (_status, { modelType, mode }) => {
            toast({
                title: "Model refresh started",
                description: `${modelType} refresh (${mode.replace(/_/g, " ")}) started.`,
            });
            queryClient.invalidateQueries({ queryKey: ["admin-model-refresh"] });
            queryClient.invalidateQueries({ queryKey: ["admin-models"] });
        },
        onError: (error: Error) => {
            toast({
                variant: "destructive",
                title: "Error",
                description: error.message,
            });
        },
    });
    const uploadModelMutation = useMutation({
        mutationFn: async ({ modelType, file }: { modelType: string; file: File }) => {
            const body = new FormData();
            body.append("archive", file);
            const res = await authFetch(`/api/admin/models/${modelType}/upload`, {
                method: "POST",
                body,
            });
            if (!res.ok) {
                const error = await res.json();
                throw new Error(error.detail || "Failed to upload model archive");
            }
            return res.json();
        },
        onSuccess: (_status, { modelType }) => {
            toast({
                title: "Model uploaded",
                description: `${modelType} archive uploaded, validated, and refreshed.`,
            });
            queryClient.invalidateQueries({ queryKey: ["admin-model-refresh"] });
            queryClient.invalidateQueries({ queryKey: ["admin-models"] });
        },
        onError: (error: Error) => {
            toast({
                variant: "destructive",
                title: "Upload failed",
                description: error.message,
            });
        },
    });
    const airgapCheckMutation = useMutation({
        mutationFn: async (): Promise<AirgapReadinessResponse> => {
            const res = await authFetch("/api/admin/models/airgap");
            if (!res.ok) {
                const error = await res.json();
                throw new Error(error.detail || "Failed to check air-gap readiness");
            }
            return res.json();
        },
        onSuccess: (data) => {
            if (data.ready) {
                toast({
                    title: "Cache ready",
                    description: "All models are cached and validated for pipeline use.",
                });
                return;
            }
            const missing = data.missing_models.join(", ") || "none";
            const invalid = data.invalid_models.join(", ") || "none";
            toast({
                variant: "destructive",
                title: "Cache not ready",
                description: `Missing: ${missing}. Invalid: ${invalid}.`,
            });
        },
        onError: (error: Error) => {
            toast({
                variant: "destructive",
                title: "Cache readiness check failed",
                description: error.message,
            });
        },
    });

    useEffect(() => {
        if (refreshError) {
            const message =
                refreshError instanceof Error
                    ? refreshError.message
                    : "Failed to fetch model refresh status.";
            if (refreshErrorRef.current === message) {
                return;
            }
            refreshErrorRef.current = message;
            toast({
                variant: "destructive",
                title: "Refresh status error",
                description: message,
            });
        }
    }, [refreshError, toast]);

    useEffect(() => {
        if (!refreshStatus) {
            return;
        }

        const previousState = refreshStateRef.current;
        refreshStateRef.current = refreshStatus.state;

        if (previousState === "running" && refreshStatus.state !== "running") {
            queryClient.invalidateQueries({ queryKey: ["admin-models"] });
        }

        if (
            previousState === "running" &&
            refreshStatus.state === "completed" &&
            refreshStatus.warnings &&
            refreshStatus.warnings.length > 0
        ) {
            toast({
                variant: "destructive",
                title: "Model readiness warning",
                description: refreshStatus.warnings.join(" | "),
            });
        }

        if (previousState !== "failed" && refreshStatus.state === "failed") {
            toast({
                variant: "destructive",
                title: "Model refresh failed",
                description: refreshStatus.error || "Warm-up did not complete.",
            });
        }
    }, [queryClient, refreshStatus, toast]);

    const refreshBusy =
        refreshMutation.isPending ||
        refreshModelMutation.isPending ||
        uploadModelMutation.isPending ||
        refreshStatus?.state === "running";

    const handleDelete = (modelType: string) => {
        if (refreshBusy) {
            return;
        }
        setDeleteModelType(modelType);
    };

    const handleModelRefresh = (modelType: string, mode: string) => {
        if (mode === "force_redownload") {
            const proceed = confirm(
                `Force re-download ${modelType}? This will delete the existing cache first.`
            );
            if (!proceed) {
                return;
            }
        }
        if (mode === "recovery") {
            const proceed = confirm(
                `Run recovery for ${modelType}? This will clean incomplete caches and re-download if invalid.`
            );
            if (!proceed) {
                return;
            }
        }
        refreshModelMutation.mutate({ modelType, mode });
    };

    const handleUploadArchiveClick = (modelType: string) => {
        setUploadTargetModel(modelType);
        if (uploadInputRef.current) {
            uploadInputRef.current.value = "";
            uploadInputRef.current.click();
        }
    };

    const handleArchiveSelected = (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file || !uploadTargetModel) {
            return;
        }
        uploadModelMutation.mutate({ modelType: uploadTargetModel, file });
    };

    const requiredPipelineNotReady =
        data?.models.filter(
            (model) => model.hard_required && model.intended_usage_ready === false
        ) || [];
    const optionalPipelineNotReady =
        data?.models.filter(
            (model) => !model.hard_required && model.intended_usage_ready === false
        ) || [];
    const modelWarnings = data?.warnings || [];

    if (isLoading) {
        return (
            <Layout>
                <div className="flex h-[50vh] items-center justify-center">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
            </Layout>
        );
    }

    if (error) {
        return (
            <Layout>
                <div className="flex h-[50vh] items-center justify-center text-destructive bg-white p-4 rounded border">
                    <AlertCircle className="mr-2 h-5 w-5" />
                    Error loading models: {error.message}
                </div>
            </Layout>
        );
    }

    return (
        <Layout>
            <div className="space-y-6">
                <input
                    ref={uploadInputRef}
                    type="file"
                    className="hidden"
                    accept=".zip,.tar,.tar.gz,.tgz"
                    onChange={handleArchiveSelected}
                />
                {requiredPipelineNotReady.length > 0 && (
                    <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-white px-4 py-3 text-sm text-destructive">
                        <AlertCircle className="mt-0.5 h-4 w-4" />
                        <span>
                            Pipeline required models are not ready:{" "}
                            <strong>
                                {requiredPipelineNotReady.map((m) => m.model_type).join(", ")}
                            </strong>
                            . Balanced/maximum capabilities may degrade until you run refresh/upload and warm-up.
                        </span>
                    </div>
                )}
                {requiredPipelineNotReady.length === 0 && optionalPipelineNotReady.length > 0 && (
                    <div className="flex items-start gap-2 rounded-md border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                        <AlertCircle className="mt-0.5 h-4 w-4" />
                        <span>
                            Optional model capabilities are currently not ready for:{" "}
                            <strong>{optionalPipelineNotReady.map((m) => m.model_type).join(", ")}</strong>.
                        </span>
                    </div>
                )}
                {modelWarnings.length > 0 && (
                    <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-white px-4 py-3 text-sm text-destructive">
                        <AlertCircle className="mt-0.5 h-4 w-4" />
                        <span>
                            Model access/download warnings:{" "}
                            <strong>{modelWarnings.join(" | ")}</strong>
                        </span>
                    </div>
                )}
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight">Model Management</h1>
                        <p className="text-muted-foreground mt-2">
                            Monitor and manage cached HuggingFace models used in the optimization pipeline.
                        </p>
                    </div>
                {refreshStatus?.state === "running" && (
                    <div className="mb-4 flex items-center gap-2 rounded-md border border-blue-200 bg-blue-50 px-4 py-2 text-sm text-blue-800">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span>
                            Model refresh ({refreshStatus.mode || "download_missing"}) is running...
                            {refreshStatus.target_models?.length
                                ? ` Targeting ${refreshStatus.target_models.join(", ")}`
                                : ""}
                        </span>
                    </div>
                )}
                <div className="flex flex-wrap items-center gap-2">
                        <ConfirmDialog
                            title="Download missing or invalid models?"
                            description="Downloads only missing or invalid models and runs warm-up."
                            confirmText="Download Missing"
                            onConfirm={() => refreshMutation.mutate("download_missing")}
                            trigger={
                                <Button variant="outline" disabled={refreshBusy}>
                                    <RefreshCw className="mr-2 h-4 w-4" />
                                    Download Missing
                                </Button>
                            }
                        />
                        <ConfirmDialog
                            title="Force re-download all models?"
                            description="Deletes existing cache directories and downloads fresh snapshots."
                            confirmText="Force Redownload"
                            onConfirm={() => refreshMutation.mutate("force_redownload")}
                            trigger={
                                <Button variant="outline" disabled={refreshBusy}>
                                    <RefreshCw className="mr-2 h-4 w-4" />
                                    Force Redownload
                                </Button>
                            }
                        />
                        <ConfirmDialog
                            title="Run recovery refresh?"
                            description="Cleans incomplete caches and re-downloads invalid models."
                            confirmText="Recovery Run"
                            onConfirm={() => refreshMutation.mutate("recovery")}
                            trigger={
                                <Button variant="outline" disabled={refreshBusy}>
                                    <RefreshCw className="mr-2 h-4 w-4" />
                                    Recovery Run
                                </Button>
                            }
                        />
                        <Button
                            variant="outline"
                            disabled={airgapCheckMutation.isPending}
                            onClick={() => airgapCheckMutation.mutate()}
                        >
                            {airgapCheckMutation.isPending ? (
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            ) : (
                                <CheckCircle2 className="mr-2 h-4 w-4" />
                            )}
                            Validate Cache Readiness
                        </Button>
                        <Button
                            onClick={() => setIsAddOpen(true)}
                            disabled={refreshBusy}
                            tooltip="Add a new model to the system"
                        >
                            <Plus className="mr-2 h-4 w-4" />
                            Add Model
                        </Button>
                    </div>
                </div>

                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                    <Card className="glass-card">
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">Total Cached Size</CardTitle>
                            <HardDrive className="h-4 w-4 text-muted-foreground" />
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">{data?.total_size_formatted || "0 B"}</div>
                        </CardContent>
                    </Card>
                    <Card className="glass-card">
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">Models Cached</CardTitle>
                            <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">
                                {data?.models.filter(m => m.cached_ok).length || 0} / {data?.models.length || 0}
                            </div>
                        </CardContent>
                    </Card>
                </div>

                <Card className="glass-card">
                    <CardHeader>
                        <CardTitle>Language Models Inventory</CardTitle>
                        <CardDescription>
                             List of all required models, metadata, and caching status.
                             <span className="mt-1 block text-xs">
                                 Model names can repeat across different model types (e.g. semantic_guard vs semantic_rank). When the HuggingFace identifier is the same, it is cached once and the runtime encoder is reused.
                             </span>
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Component</TableHead>
                                    <TableHead>Model Name</TableHead>
                                    <TableHead>Revision</TableHead>
                                    <TableHead>Library / Type</TableHead>
                                    <TableHead>Usage</TableHead>
                                    <TableHead>Size</TableHead>
                                    <TableHead>Cached Status</TableHead>
                                    <TableHead>Loaded Status</TableHead>
                                    <TableHead>Intended Pipeline Usage</TableHead>
                                    <TableHead>Last Refresh</TableHead>
                                    <TableHead className="text-right">Actions</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {data?.models.map((model) => (
                                    <TableRow key={model.model_type}>
                                        <TableCell className="font-medium">
                                            <div className="flex flex-col">
                                                <span>{model.component || model.model_type}</span>
                                                <span className="text-xs text-muted-foreground">
                                                    {model.model_type}
                                                </span>
                                            </div>
                                        </TableCell>
                                        <TableCell className="font-mono text-xs">
                                            {model.model_name}
                                        </TableCell>
                                        <TableCell className="font-mono text-xs">
                                            {model.revision || "default"}
                                        </TableCell>
                                        <TableCell>{model.library_type || "N/A"}</TableCell>
                                        <TableCell>{model.usage || "N/A"}</TableCell>
                                        <TableCell>{model.size_formatted}</TableCell>
                                        <TableCell>
                                            <div className="flex flex-col gap-1">
                                                {model.cached_ok ? (
                                                    <Badge
                                                        variant="default"
                                                        className="bg-green-500/15 text-green-500 hover:bg-green-500/25 border-green-500/20"
                                                    >
                                                        Cached OK
                                                    </Badge>
                                                ) : (
                                                    <Badge variant="destructive">
                                                        Missing/Invalid
                                                    </Badge>
                                                )}
                                                {model.cached_reason && (
                                                    <span className="text-xs text-muted-foreground">
                                                        {cachedReasonLabels[model.cached_reason] ?? model.cached_reason}
                                                    </span>
                                                )}
                                                {model.cached_error_detail && (
                                                    <span className="text-xs text-destructive/90">
                                                        {model.cached_error_detail}
                                                    </span>
                                                )}
                                            </div>
                                        </TableCell>
                                        <TableCell>
                                            <div className="flex flex-col gap-1">
                                                {model.loaded_ok === true ? (
                                                    <Badge
                                                        variant="default"
                                                        className="bg-green-500/15 text-green-500 hover:bg-green-500/25 border-green-500/20"
                                                    >
                                                        Loaded OK
                                                    </Badge>
                                                ) : (
                                                    <Badge variant="secondary">
                                                        {getLoadedStatusBadgeLabel(model)}
                                                    </Badge>
                                                )}
                                                {model.loaded_reason && (
                                                    <span className="text-xs text-muted-foreground">
                                                        {loadedReasonLabels[model.loaded_reason] ?? model.loaded_reason}
                                                    </span>
                                                )}
                                            </div>
                                        </TableCell>
                                        <TableCell>
                                            <div className="flex flex-col gap-1">
                                                {model.intended_usage_ready ? (
                                                    <Badge
                                                        variant="default"
                                                        className="bg-emerald-500/15 text-emerald-500 hover:bg-emerald-500/25 border-emerald-500/20"
                                                    >
                                                        {getIntendedUsageBadgeLabel(model)}
                                                    </Badge>
                                                ) : (
                                                    <Badge variant="secondary">
                                                        {getIntendedUsageBadgeLabel(model)}
                                                    </Badge>
                                                )}
                                                <span className="text-xs text-muted-foreground">
                                                    {(model.intended_features && model.intended_features.length > 0)
                                                        ? model.intended_features.join(", ")
                                                        : "No declared intended features"}
                                                </span>
                                            </div>
                                        </TableCell>
                                        <TableCell>
                                            {model.last_refresh || "N/A"}
                                        </TableCell>
                                        <TableCell className="text-right space-x-2">
                                            <DropdownMenu>
                                                <DropdownMenuTrigger asChild>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        disabled={refreshBusy}
                                                        aria-label={`Refresh ${model.model_type}`}
                                                        tooltip="Refresh this model"
                                                    >
                                                        <RefreshCw className="h-4 w-4 stroke-3 text-sidebar-primary" />
                                                    </Button>
                                                </DropdownMenuTrigger>
                                                <DropdownMenuContent align="end">
                                                    <DropdownMenuItem
                                                        onClick={() =>
                                                            handleModelRefresh(
                                                                model.model_type,
                                                                "download_missing"
                                                            )
                                                        }
                                                    >
                                                        Download Missing
                                                    </DropdownMenuItem>
                                                    <DropdownMenuItem
                                                        onClick={() =>
                                                            handleModelRefresh(
                                                                model.model_type,
                                                                "force_redownload"
                                                            )
                                                        }
                                                    >
                                                        Force Redownload
                                                    </DropdownMenuItem>
                                                    <DropdownMenuItem
                                                        onClick={() =>
                                                            handleModelRefresh(
                                                                model.model_type,
                                                                "recovery"
                                                            )
                                                        }
                                                    >
                                                        Recovery Run
                                                    </DropdownMenuItem>
                                                    {model.model_type !== "spacy" && (
                                                        <DropdownMenuItem
                                                            onClick={() => handleUploadArchiveClick(model.model_type)}
                                                        >
                                                            <Upload className="mr-2 h-4 w-4" />
                                                            Upload Archive
                                                        </DropdownMenuItem>
                                                    )}
                                                </DropdownMenuContent>
                                            </DropdownMenu>
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                disabled={refreshBusy}
                                                aria-label={`Edit ${model.model_type}`}
                                                onClick={() => setEditingModel(model)}
                                                tooltip="Edit this model"
                                            >
                                                <Edit2 className="h-4 w-4 stroke-3 text-sidebar-primary" />
                                            </Button>
                                            <Button 
                                                variant="ghost" 
                                                size="icon" 
                                                disabled={refreshBusy}
                                                aria-label={`Delete ${model.model_type}`}
                                                onClick={() => handleDelete(model.model_type)}
                                                tooltip="Delete this model"
                                            >
                                                <Trash2 className="h-4 w-4 stroke-3 text-sidebar-primary" />
                                            </Button>
                                        </TableCell>
                                    </TableRow>
                                ))}
                                {(!data?.models || data.models.length === 0) && (
                                    <TableRow>
                                        <TableCell colSpan={11} className="text-center py-8 text-muted-foreground">
                                            No models found.
                                        </TableCell>
                                    </TableRow>
                                )}
                            </TableBody>
                        </Table>
                    </CardContent>
                </Card>
            </div>
            
            {editingModel && (
                <EditModelDialog 
                    model={editingModel} 
                    open={!!editingModel} 
                    onOpenChange={(open) => !open && setEditingModel(null)} 
                />
            )}
            
            <AddModelDialog 
                open={isAddOpen}
                onOpenChange={setIsAddOpen}
            />

            {deleteModelType && (
                <ConfirmDialog
                    title="Delete Model Configuration"
                    description={
                        (() => {
                            const effectiveProtected = new Set(
                                protectedModelTypes?.protected_model_types?.length
                                    ? protectedModelTypes.protected_model_types
                                    : Array.from(PROTECTED_MODEL_TYPES),
                            );
                            const isProtected = effectiveProtected.has(deleteModelType);
                            return isProtected
                                ? `Deleting ${deleteModelType} will remove a core optimizer dependency and may break semantic/entropy/token classifier flows. This action cannot be undone.`
                                : `Are you sure you want to delete the configuration for ${deleteModelType}? This action cannot be undone.`;
                        })()
                    }
                    confirmText="Delete"
                    onConfirm={() => {
                        const effectiveProtected = new Set(
                            protectedModelTypes?.protected_model_types?.length
                                ? protectedModelTypes.protected_model_types
                                : Array.from(PROTECTED_MODEL_TYPES),
                        );
                        const isProtected = effectiveProtected.has(deleteModelType);
                        deleteMutation.mutate({ modelType: deleteModelType, overrideCore: isProtected });
                        setDeleteModelType(null);
                    }}
                    onCancel={() => setDeleteModelType(null)}
                    open={!!deleteModelType}
                    onOpenChange={(open) => !open && setDeleteModelType(null)}
                />
            )}
        </Layout>
    );
}
