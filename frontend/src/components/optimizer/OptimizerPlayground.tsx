import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Zap, Copy, RefreshCw, HelpCircle, ArrowRightLeft, Trash2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Input } from "@/components/ui/input";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ToastAction } from "@/components/ui/toast";
import { authFetch } from "@/lib/authFetch";

interface FieldLabelProps {
  label: string;
  help: string;
  labelClassName?: string;
}

function FieldLabel({ label, help, labelClassName }: FieldLabelProps) {
  return (
    <div className="flex items-center gap-2">
      <span
        className={
          labelClassName ?? "text-xs font-medium text-muted-foreground uppercase tracking-wider"
        }
      >
        {label}
      </span>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            className="text-muted-foreground/70 hover:text-muted-foreground transition-colors"
            aria-label={`${label} help`}
          >
            <HelpCircle className="h-3.5 w-3.5" />
          </button>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs">
          {help}
        </TooltipContent>
      </Tooltip>
    </div>
  );
}

interface OptimizationApiResponse {
  optimized_output: string;
  stats: {
    original_tokens: number;
    optimized_tokens: number;
    token_savings: number;
    compression_percentage: number;
    processing_time_ms: number;
    semantic_similarity?: number | null;
    content_profile: string;
    smart_context_description: string;
    deduplication?: Record<string, number>;
  };
  router?: {
    content_type: string;
    profile: string;
  } | null;
  techniques_applied?: string[];
  warnings?: string[] | null;
}

interface LLMProviderOption {
  value: string;
  label: string;
}

interface LLMProviderInfo {
  key: string;
  label: string;
  models: LLMProviderOption[];
}

interface LLMProviderListResponse {
  providers: LLMProviderInfo[];
}

interface LLMTestResponse {
  text: string;
  duration_ms: number;
}

interface LLMTestTiming {
  wall_ms: number;
}

interface LLMProfile {
  name: string;
  provider: string;
  model: string;
  has_api_key: boolean;
}

interface SettingsResponse {
  llm_profiles: LLMProfile[];
}

export function OptimizerPlayground() {
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [optimizationTechnique, setOptimizationTechnique] = useState<"rule_based" | "llm_based">("rule_based");
  const [optimizationMode, setOptimizationMode] = useState<"conservative" | "balanced" | "maximum">("balanced");
  const [batchName, setBatchName] = useState("");
  const [segmentSpans, setSegmentSpans] = useState("");
  const [segmentSpansError, setSegmentSpansError] = useState("");
  const [queryHint, setQueryHint] = useState("");
  const [customCanonicals, setCustomCanonicals] = useState("");
  const [customCanonicalsError, setCustomCanonicalsError] = useState("");
  const [additionalOpen, setAdditionalOpen] = useState(false);
  const [stats, setStats] = useState<OptimizationApiResponse["stats"] | null>(null);
  const [routerInfo, setRouterInfo] = useState<OptimizationApiResponse["router"] | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [lastOptimizedInput, setLastOptimizedInput] = useState("");
  const [techniques, setTechniques] = useState<string[]>([]);
  const [llmProvider, setLlmProvider] = useState("");
  const [llmModel, setLlmModel] = useState("");
  const [llmCustomModel, setLlmCustomModel] = useState("");
  const [llmApiKey, setLlmApiKey] = useState("");
  const [llmProfileName, setLlmProfileName] = useState("manual");
  const [llmTestingOpen, setLlmTestingOpen] = useState(false);
  const [originalTestResult, setOriginalTestResult] = useState<LLMTestResponse | null>(null);
  const [optimizedTestResult, setOptimizedTestResult] = useState<LLMTestResponse | null>(null);
  const [originalTestTiming, setOriginalTestTiming] = useState<LLMTestTiming | null>(null);
  const [optimizedTestTiming, setOptimizedTestTiming] = useState<LLMTestTiming | null>(null);
  const { toast } = useToast();

  const { data: llmProviders } = useQuery<LLMProviderListResponse>({
    queryKey: ["llm-providers"],
    queryFn: async () => {
      const response = await authFetch("/api/v1/llm/providers");
      if (!response.ok) {
        throw new Error("Failed to load LLM providers");
      }
      return response.json();
    },
  });

  const { data: settings } = useQuery<SettingsResponse>({
    queryKey: ["settings"],
    queryFn: async () => {
      const response = await authFetch("/api/v1/settings");
      if (!response.ok) {
        throw new Error("Failed to load settings");
      }
      return response.json();
    },
  });

  const llmProfiles = settings?.llm_profiles ?? [];
  const providerOptions = llmProviders?.providers ?? [];
  const activeProvider = providerOptions.find((provider) => provider.key === llmProvider) ?? providerOptions[0];
  const modelOptions = activeProvider?.models ?? [];
  const hasProviders = providerOptions.length > 0;
  const hasModels = modelOptions.length > 0;
  const selectedModel = llmModel || modelOptions[0]?.value || "";
  const resolvedModel = selectedModel === "other" ? llmCustomModel.trim() : selectedModel;

  const estimatedInputTokens = input.length > 0 ? Math.ceil(input.length / 4) : 0;
  const estimatedOutputTokens = output.length > 0 ? Math.ceil(output.length / 4) : 0;
  const displayedInputTokens =
    stats && lastOptimizedInput === input ? stats.original_tokens : estimatedInputTokens;
  const displayedOutputTokens = stats ? stats.optimized_tokens : estimatedOutputTokens;
  useEffect(() => {
    if (llmProfileName === "manual" || !llmProfiles.length) return;
    const profile = llmProfiles.find((item) => item.name === llmProfileName);
    if (!profile) return;
    setLlmProvider(profile.provider);
    const provider = providerOptions.find((item) => item.key === profile.provider);
    const hasModel = provider?.models?.some((model) => model.value === profile.model);
    if (hasModel) {
      setLlmModel(profile.model);
      setLlmCustomModel("");
    } else {
      setLlmModel("other");
      setLlmCustomModel(profile.model);
    }
    setLlmApiKey("");
  }, [llmProfileName, llmProfiles, providerOptions]);

  useEffect(() => {
    if (!providerOptions.length) return;
    const hasMatch = providerOptions.some((provider) => provider.key === llmProvider);
    if (!hasMatch) {
      setLlmProvider(providerOptions[0].key);
    }
  }, [providerOptions, llmProvider]);

  useEffect(() => {
    if (!activeProvider?.models?.length) return;
    const modelMatch = activeProvider.models.some((model) => model.value === llmModel);
    if (!llmModel || !modelMatch) {
      setLlmModel(activeProvider.models[0].value);
    }
  }, [activeProvider, llmModel]);

  const optimizeMutation = useMutation({
    mutationFn: async (prompt: string) => {
      const payload: Record<string, unknown> = {
        prompt,
        optimization_technique: optimizationTechnique,
        optimization_mode: optimizationMode,
      };

      const trimmedBatchName = batchName.trim();
      if (trimmedBatchName) payload.name = trimmedBatchName;

      const trimmedQuery = queryHint.trim();
      if (trimmedQuery) payload.query = trimmedQuery;

      if (segmentSpans.trim()) {
        let parsed: unknown;
        try {
          parsed = JSON.parse(segmentSpans);
        } catch (error) {
          const message =
            error instanceof Error ? error.message : "Segment spans must be valid JSON.";
          throw new Error(message);
        }
        if (!Array.isArray(parsed)) {
          throw new Error("Segment spans must be a JSON array.");
        }
        payload.segment_spans = parsed;
      }

      if (customCanonicals.trim()) {
        let parsed: unknown;
        try {
          parsed = JSON.parse(customCanonicals);
        } catch (error) {
          const message =
            error instanceof Error ? error.message : "Custom canonicals must be valid JSON.";
          throw new Error(message);
        }
        if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
          throw new Error("Custom canonicals must be a JSON object.");
        }
        const entries = Object.entries(parsed as Record<string, unknown>);
        const invalidEntry = entries.find(
          ([key, value]) => !key.trim() || typeof value !== "string" || !value.trim()
        );
        if (invalidEntry) {
          throw new Error("Custom canonicals must map non-empty strings to non-empty strings.");
        }
        payload.custom_canonicals = parsed;
      }

      const url = "/api/v1/optimize";
      const response = await authFetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        let detail = "Optimization failed";
        try {
          const errorBody = await response.json() as { detail?: string };
          if (typeof errorBody?.detail === "string" && errorBody.detail.trim()) {
            detail = errorBody.detail;
          }
        } catch {
          // keep default message when response body is not JSON
        }
        throw new Error(detail);
      }

      return response.json() as Promise<OptimizationApiResponse>;
    },
    onSuccess: (data) => {
      setOutput(data.optimized_output);
      setStats(data.stats);
      setRouterInfo(data.router ?? null);
      setWarnings((data.warnings ?? []).filter(Boolean));
      setTechniques(data.techniques_applied || []);
      toast({
        title: "Optimization Complete",
        description: `Reduced token count by ${data.stats.compression_percentage.toFixed(1)}%`,
        action: (
          <ToastAction
            altText="Copy output"
            onClick={() => {
              navigator.clipboard.writeText(data.optimized_output);
              toast({ title: "Copied to clipboard" });
            }}
          >
            Copy
          </ToastAction>
        ),
      });
    },
    onError: (error) => {
      const rawMessage = error instanceof Error ? error.message : "";
      const message =
        rawMessage && /failed to fetch|networkerror|network request/i.test(rawMessage)
          ? "Network/API timeout while contacting LLM optimizer. Try a shorter prompt, retry, or verify backend CORS/API health."
          : rawMessage || "Please try again with a different prompt.";
      toast({
        title: "Optimization Failed",
        description: message,
        variant: "destructive",
      });
    },
  });

  const llmTestRequest = async (prompt: string) => {
    const response = await authFetch("/api/v1/llm/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: activeProvider?.key ?? "",
        model: resolvedModel,
        prompt,
        api_key: llmApiKey.trim() ? llmApiKey.trim() : undefined,
        profile_name: llmProfileName !== "manual" ? llmProfileName : undefined,
      }),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "LLM test failed");
    }
    return response.json() as Promise<LLMTestResponse>;
  };

  const originalTestMutation = useMutation({
    mutationFn: llmTestRequest,
    onError: (error) => {
      toast({
        title: "Original LLM Test Failed",
        description: error.message || "Check your provider, model, and API key.",
        variant: "destructive",
      });
    },
  });

  const optimizedTestMutation = useMutation({
    mutationFn: llmTestRequest,
    onError: (error) => {
      toast({
        title: "Optimized LLM Test Failed",
        description: error.message || "Check your provider, model, and API key.",
        variant: "destructive",
      });
    },
  });

  const runTimedTest = async (
    prompt: string,
    mutation: typeof originalTestMutation,
    setResult: (value: LLMTestResponse | null) => void,
    setTiming: (value: LLMTestTiming | null) => void
  ) => {
    const start = performance.now();
    try {
      const result = await mutation.mutateAsync(prompt);
      setResult(result);
      return result;
    } finally {
      setTiming({ wall_ms: performance.now() - start });
    }
  };

  const handleOptimize = () => {
    if (!input) return;
    setLastOptimizedInput(input);
    setWarnings([]);
    setRouterInfo(null);
    optimizeMutation.mutate(input);
  };

  const validateSegmentSpans = (value: string): string => {
    if (!value.trim()) {
      return "";
    }

    try {
      const parsed = JSON.parse(value);
      if (!Array.isArray(parsed)) {
        return "Segment spans must be a JSON array";
      }
      return "";
    } catch (error) {
      if (error instanceof SyntaxError) {
        return "Invalid JSON syntax";
      }
      return "Invalid JSON";
    }
  };

  const handleSegmentSpansChange = (value: string) => {
    setSegmentSpans(value);
    const error = validateSegmentSpans(value);
    setSegmentSpansError(error);
  };

  const validateCustomCanonicals = (value: string): string => {
    if (!value.trim()) {
      return "";
    }
    try {
      const parsed = JSON.parse(value);
      if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
        return "Custom canonicals must be a JSON object";
      }
      const entries = Object.entries(parsed as Record<string, unknown>);
      if (
        entries.some(
          ([key, val]) => !key.trim() || typeof val !== "string" || !val.trim()
        )
      ) {
        return "Custom canonicals require non-empty string keys and values";
      }
      return "";
    } catch (error) {
      if (error instanceof SyntaxError) {
        return "Invalid JSON syntax";
      }
      return "Invalid JSON";
    }
  };

  const handleCustomCanonicalsChange = (value: string) => {
    setCustomCanonicals(value);
    const error = validateCustomCanonicals(value);
    setCustomCanonicalsError(error);
  };

  const resetRunState = () => {
    setOutput("");
    setStats(null);
    setRouterInfo(null);
    setWarnings([]);
    setTechniques([]);
    setLastOptimizedInput("");
    setOriginalTestResult(null);
    setOptimizedTestResult(null);
    setOriginalTestTiming(null);
    setOptimizedTestTiming(null);
  };

  const handleClear = () => {
    setInput("");
    resetRunState();
  };

  const handleSwap = () => {
    if (!output) return;
    const previousInput = input;
    setInput(output);
    setOutput(previousInput);
    setStats(null);
    setRouterInfo(null);
    setWarnings([]);
    setTechniques([]);
    setLastOptimizedInput("");
    setOriginalTestResult(null);
    setOptimizedTestResult(null);
    setOriginalTestTiming(null);
    setOptimizedTestTiming(null);
  };

  const isOllama = activeProvider?.key === "ollama";
  const selectedProfile =
    llmProfileName !== "manual"
      ? llmProfiles.find((profile) => profile.name === llmProfileName) ?? null
      : null;
  const canRunLlmTest = Boolean(
    activeProvider?.key &&
    resolvedModel &&
    (isOllama || llmApiKey.trim() || Boolean(selectedProfile?.has_api_key))
  );
  const isTestPending = originalTestMutation.isPending || optimizedTestMutation.isPending;
  const handleTestOriginal = async () => {
    if (!input || !canRunLlmTest) return;
    setOriginalTestResult(null);
    setOriginalTestTiming(null);
    try {
      await runTimedTest(input, originalTestMutation, setOriginalTestResult, setOriginalTestTiming);
    } catch {
      setOriginalTestResult(null);
    }
  };

  const handleTestOptimized = async () => {
    if (!output || !canRunLlmTest) return;
    setOptimizedTestResult(null);
    setOptimizedTestTiming(null);
    try {
      await runTimedTest(output, optimizedTestMutation, setOptimizedTestResult, setOptimizedTestTiming);
    } catch {
      setOptimizedTestResult(null);
    }
  };

  const handleTestBoth = async () => {
    if (!input || !output || !canRunLlmTest) return;
    setOriginalTestResult(null);
    setOptimizedTestResult(null);
    setOriginalTestTiming(null);
    setOptimizedTestTiming(null);
    await Promise.allSettled([
      runTimedTest(output, optimizedTestMutation, setOptimizedTestResult, setOptimizedTestTiming),
      runTimedTest(input, originalTestMutation, setOriginalTestResult, setOriginalTestTiming),
    ]);
  };

  return (
    <div className="grid grid-rows-[auto_1fr_auto] gap-6 min-h-[calc(100vh-12rem)]">
      {/* Top Row: Controls */}
      <div className="w-full bg-card shadow-sm p-4 rounded-xl border border-border backdrop-blur-sm">
        {/* Warnings */}
        {warnings.length > 0 && (
          <div className="mb-4 rounded border border-amber-500/40 bg-amber-500/10 px-3 py-1.5 flex items-center gap-3">
            <div className="text-[10px] font-bold text-amber-200 uppercase tracking-wider shrink-0">Warnings</div>
            <div className="text-xs text-amber-100/90 truncate flex-1">
              {warnings.join(" • ")}
            </div>
          </div>
        )}

        <div className="flex flex-col lg:flex-row gap-6 items-start">
          {/* Left Side: Mode, Stats, Content Type (60%) */}
          <div className="w-full lg:w-[60%] space-y-3">
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
              <div className="space-y-1.5">
                <FieldLabel
                  label="Optimization Technique"
                  help="Choose between local rule-based optimization and advanced LLM-based optimization."
                />
                <Select
                  value={optimizationTechnique}
                  onValueChange={(v) => setOptimizationTechnique(v as typeof optimizationTechnique)}
                >
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue placeholder="Select technique" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="rule_based">Rule based (FAST)</SelectItem>
                    <SelectItem value="llm_based">LLM based (ADVANCED)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <FieldLabel
                  label="Optimization Mode"
                  help="Controls how aggressive the optimizer is. Conservative keeps meaning safest, balanced is the default, maximum enables deeper passes for more savings."
                />
                <Select
                  value={optimizationMode}
                  onValueChange={(v) => setOptimizationMode(v as typeof optimizationMode)}
                >
                  <SelectTrigger data-testid="select-mode" className="h-8 text-xs">
                    <SelectValue placeholder="Select mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="conservative">Conservative (Lossless)</SelectItem>
                    <SelectItem value="balanced">Balanced (Recommended)</SelectItem>
                    <SelectItem value="maximum">Maximum (Best Savings)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <FieldLabel
                  label="Semantic Similarity"
                  help="Approximate similarity between the original and optimized prompt. Higher is closer in meaning."
                />
                {stats ? (
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs">
                      <span className="text-primary font-mono" data-testid="text-similarity">
                        {((stats.semantic_similarity ?? 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary transition-all"
                        style={{ width: `${(stats.semantic_similarity ?? 0) * 100}%` }}
                      />
                    </div>
                  </div>
                ) : (
                  <div className="h-8 bg-secondary/20 rounded flex items-center justify-center text-xs text-muted-foreground">
                    Run optimization
                  </div>
                )}
              </div>

              <div className="space-y-1.5">
                <FieldLabel
                  label="Tokens Saved"
                  help="Estimated token reduction based on the optimized output length."
                />
                {stats ? (
                  <div className="h-8 flex items-center">
                    <span className="font-mono text-green-400 text-base font-semibold" data-testid="text-tokens-saved">{stats.token_savings}</span>
                    <span className="text-xs text-muted-foreground ml-1.5">(-{stats.compression_percentage.toFixed(0)}%)</span>
                  </div>
                ) : (
                  <div className="h-8 bg-secondary/20 rounded flex items-center justify-center text-xs text-muted-foreground">
                    -
                  </div>
                )}
              </div>
            </div>

            {stats?.content_profile && (
              <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground mt-1">
                <FieldLabel
                  label="Detected Content Type"
                  help="Smart selection is active and adjusts technical defaults based on the detected content."
                  labelClassName="text-[10px] font-medium text-muted-foreground uppercase tracking-wider"
                />
                <Badge variant="outline" className="text-[10px] font-mono h-5 px-1.5" data-testid="badge-content-type">
                  {stats.content_profile}
                </Badge>
                {stats.smart_context_description && (
                  <span className="text-[10px] text-muted-foreground/80">{stats.smart_context_description}</span>
                )}
              </div>
            )}
          </div>

          {/* Right Side: Active Algorithms (40%) */}
          <div className="w-full lg:w-[40%] lg:border-l lg:border-border/50 lg:pl-6 min-h-[80px]">
            <FieldLabel
              label="Active Algorithms"
              help="Optimization techniques that were applied in the current run."
            />
            {techniques.length > 0 ? (
              <div className="flex flex-wrap gap-1.5 mt-2">
                {techniques.map((tech) => (
                  <Badge key={tech} variant="secondary" className="text-[10px] h-5 px-1.5">{tech}</Badge>
                ))}
              </div>
            ) : (
              <div className="mt-2 text-[10px] text-muted-foreground italic">
                Ready to optimize.
              </div>
            )}
          </div>
        </div>

        <Accordion
          type="single"
          collapsible
          value={additionalOpen ? "additional-options" : ""}
          onValueChange={(value) => setAdditionalOpen(value === "additional-options")}
          className="mt-4"
        >
          <AccordionItem value="additional-options" className="border-2 border-border bg-card/30 rounded-lg px-4 shadow-sm">
            <AccordionTrigger className="text-sm font-medium">Additional Controls (Optional)</AccordionTrigger>
            <AccordionContent>
              <p className="text-xs text-muted-foreground">
                Optional inputs for preserving critical spans or query-aware context. Leave empty for maximum compression.
              </p>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-2">
                <Card className="p-4 space-y-3">
                  <FieldLabel
                    label="Batch Name"
                    help="Optional run label used in dashboards/history when applicable."
                  />
                  <Input
                    placeholder="e.g. onboarding-optimizations"
                    value={batchName}
                    onChange={(e) => setBatchName(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Optional and safe to leave blank for single prompt runs.
                  </p>
                </Card>

                <Card className="p-4 space-y-3">
                  <FieldLabel
                    label="Query Hint (RAG only)"
                    help="Provide a query to prioritize query-relevant context during compression."
                  />
                  <Input
                    placeholder="Ask a question or provide a retrieval hint..."
                    value={queryHint}
                    onChange={(e) => setQueryHint(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Leave blank for non-RAG prompts.
                  </p>
                </Card>

                <Card className="p-4 space-y-3">
                  <FieldLabel
                    label="Segment Spans"
                    help="Provide character spans to preserve or weight during compression."
                  />
                  <Textarea
                    placeholder='[{"start": 0, "end": 42, "weight": 1.0}]'
                    value={segmentSpans}
                    onChange={(e) => handleSegmentSpansChange(e.target.value)}
                    className={`min-h-[140px] font-mono text-xs ${segmentSpansError ? "border-red-500" : ""}`}
                  />
                  {segmentSpansError && (
                    <p className="text-xs text-red-500 bg-white p-2 rounded border">{segmentSpansError}</p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Use JSON spans with optional <code>label</code> or <code>weight</code>. You can also
                    tag critical text using <code>&lt;protect&gt;...&lt;/protect&gt;</code> in the prompt.
                  </p>
                </Card>

                <Card className="p-4 space-y-3">
                  <FieldLabel
                    label="Custom Canonicals"
                    help="Per-request canonical mappings (long form to short form) as JSON."
                  />
                  <Textarea
                    placeholder='{"Natural Language Processing":"NLP","Machine Learning":"ML"}'
                    value={customCanonicals}
                    onChange={(event) => handleCustomCanonicalsChange(event.target.value)}
                    className={`min-h-[140px] font-mono text-xs ${customCanonicalsError ? "border-red-500" : ""}`}
                  />
                  {customCanonicalsError && (
                    <p className="text-xs text-red-500 bg-white p-2 rounded border">{customCanonicalsError}</p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Keys and values must be non-empty strings.
                  </p>
                </Card>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>

      <div className="flex justify-center -my-2 z-10 px-4 sm:px-0">
        <Button
          data-testid="button-optimize"
          className="h-8 w-full text-lg font-bold shadow-lg hover:shadow-xl hover:-translate-y-0.5 transition-all"
          onClick={handleOptimize}
          disabled={optimizeMutation.isPending || !input}
          tooltip={
            optimizeMutation.isPending
              ? "Optimizing your prompt..."
              : "Optimize your prompt for token savings and semantic preservation"
          }
        >
          {optimizeMutation.isPending ? (
            <>
              <RefreshCw className="mr-2 h-5 w-5 animate-spin" /> Optimizing...
            </>
          ) : (
            <>
              <Zap className="mr-2 h-5 w-5 fill-current" /> OPTIMIZE
            </>
          )}
        </Button>
      </div>

      {/* Bottom Row: Input & Output Side by Side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 lg:gap-6 h-full min-h-[495px]">
        {/* Left Column: Input */}
        <div className="flex flex-col gap-3 lg:gap-4 h-full min-h-[468px]">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 sm:gap-0 p-3 bg-muted/40 rounded-t-lg border-b border-border">
            <FieldLabel
              label="INPUT PROMPT"
              help="Paste the prompt you want to optimize. The output appears on the right."
              labelClassName="text-xs sm:text-sm font-medium text-foreground font-mono"
            />
            <div className="flex items-center gap-1 lg:gap-2">
              <Badge variant="outline" className="font-mono text-[10px] sm:text-xs bg-background" data-testid="badge-input-tokens">
                {displayedInputTokens} Tokens
              </Badge>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={handleSwap}
                disabled={!output}
                aria-label="Swap input and output"
                className="h-8 w-8 p-0 hidden lg:flex bg-background"
                tooltip="Swap input and output prompts"
              >
                <ArrowRightLeft className="h-4 w-4" />
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={handleClear}
                disabled={!input && !output}
                aria-label="Clear prompts"
                className="h-8 w-8 p-0 bg-background"
                tooltip="Clear both input and output prompts"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
          <Card className="flex-1 p-0 overflow-hidden border-2 border-border bg-card rounded-lg shadow-md hover:shadow-lg transition-all">
            <Textarea
              data-testid="textarea-prompt-input"
              placeholder="Paste your prompt here to optimize..."
              className="h-full w-full resize-none border-none focus-visible:ring-0 p-3 lg:p-4 font-mono text-xs sm:text-sm bg-transparent"
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
          </Card>
        </div>

        {/* Right Column: Output */}
        <div className="flex flex-col gap-3 lg:gap-4 h-full min-h-[468px]">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 sm:gap-0 p-3 bg-green-500/10 rounded-t-lg border-b border-border">
            <FieldLabel
              label="OPTIMIZED OUTPUT"
              help="Read-only optimized prompt. Use the copy button to copy the result."
              labelClassName="text-xs sm:text-sm font-medium text-foreground font-mono"
            />
            <div className="flex items-center gap-1 lg:gap-2 flex-wrap">
              {output && stats && (
                <Badge
                  data-testid="badge-savings"
                  variant="default"
                  className="bg-green-500 text-white border-green-600 hover:bg-green-600 font-mono text-[10px] sm:text-xs shadow-sm"
                >
                  -{stats.compression_percentage.toFixed(0)}% SAVED
                </Badge>
              )}
              <Badge variant="outline" className="font-mono text-[10px] sm:text-xs bg-background" data-testid="badge-output-tokens">
                {displayedOutputTokens} Tokens
              </Badge>
            </div>
          </div>
          <Card className="flex-1 p-0 overflow-hidden relative group bg-card border-2 border-border rounded-lg shadow-md hover:shadow-lg transition-all">
            <Textarea
              data-testid="textarea-optimized-output"
              readOnly
              className="h-full w-full resize-none border-none focus-visible:ring-0 p-3 lg:p-4 font-mono text-xs sm:text-sm bg-transparent text-primary/90"
              value={output}
              placeholder="Optimized output will appear here..."
            />
            {output && (
              <Button
                data-testid="button-copy"
                size="sm"
                variant="secondary"
                className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity h-8 w-8 p-0 shadow-sm"
                onClick={() => {
                  navigator.clipboard.writeText(output);
                  toast({ title: "Copied to clipboard" });
                }}
                tooltip="Copy optimized prompt to clipboard"
              >
                <Copy className="h-4 w-4" />
              </Button>
            )}
          </Card>

          {routerInfo && (
             <div className="flex items-center gap-3 mt-1 px-1">
               <FieldLabel
                 label="SMART ROUTER"
                 help="Detected content type and profile used to select safe optimization behaviors."
                 labelClassName="text-[10px] font-bold text-muted-foreground uppercase tracking-wider"
               />
               <div className="flex flex-wrap gap-1.5">
                 <Badge variant="outline" className="text-[10px] font-mono h-5 px-1.5">
                   {routerInfo.content_type}
                 </Badge>
                 <Badge variant="outline" className="text-[10px] font-mono h-5 px-1.5">
                   {routerInfo.profile}
                 </Badge>
               </div>
             </div>
          )}
        </div>
      </div>
      <Accordion
        type="single"
        collapsible
        value={llmTestingOpen ? "llm-testing" : ""}
        onValueChange={(value) => setLlmTestingOpen(value === "llm-testing")}
      >
        <AccordionItem value="llm-testing" className="border border-border/50 rounded-lg px-3 lg:px-4">
          <AccordionTrigger className="text-xs sm:text-sm font-medium">LLM Response Testing</AccordionTrigger>
          <AccordionContent>
            <div className="space-y-4 lg:space-y-6 p-4 bg-card/20 rounded-lg border border-border/50">
              <p className="text-xs sm:text-sm text-muted-foreground">
                Send the original and optimized prompts to a provider to compare responses.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 lg:gap-4">
                <div className="space-y-2">
                  <FieldLabel
                    label="Saved Profile"
                    help="Reuse a saved provider/model/API key combo, or choose manual entry."
                  />
                  <Select value={llmProfileName} onValueChange={(value) => setLlmProfileName(value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Manual entry" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="manual">Manual entry</SelectItem>
                      {llmProfiles.map((profile) => (
                        <SelectItem key={profile.name} value={profile.name}>
                          {profile.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <FieldLabel
                    label="Provider"
                    help="Select the LLM provider used to compare responses."
                  />
                  <Select
                    value={activeProvider?.key ?? ""}
                    onValueChange={(value) => setLlmProvider(value)}
                    disabled={!hasProviders}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select provider" />
                    </SelectTrigger>
                    <SelectContent>
                      {hasProviders ? (
                        providerOptions.map((provider) => (
                          <SelectItem key={provider.key} value={provider.key}>
                            {provider.label}
                          </SelectItem>
                        ))
                      ) : (
                        <SelectItem value="no-providers" disabled>
                          No providers available
                        </SelectItem>
                      )}
                    </SelectContent>
                  </Select>
                  {!hasProviders && (
                    <p className="text-xs text-muted-foreground">Add providers in backend settings.</p>
                  )}
                </div>
                <div className="space-y-2">
                  <FieldLabel
                    label="Model"
                    help="Choose a model from the provider list or enter a custom name."
                  />
                  <Select
                    value={selectedModel}
                    onValueChange={(value) => setLlmModel(value)}
                    disabled={!hasModels}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select model" />
                    </SelectTrigger>
                    <SelectContent>
                      {hasModels ? (
                        modelOptions.map((model) => (
                          <SelectItem key={model.value} value={model.value}>
                            {model.label}
                          </SelectItem>
                        ))
                      ) : (
                        <SelectItem value="no-models" disabled>
                          No models available
                        </SelectItem>
                      )}
                    </SelectContent>
                  </Select>
                  {!hasModels && (
                    <p className="text-xs text-muted-foreground">Pick a provider to load models.</p>
                  )}
                  {selectedModel === "other" && (
                    <Input
                      placeholder="Custom model name"
                      value={llmCustomModel}
                      onChange={(event) => setLlmCustomModel(event.target.value)}
                    />
                  )}
                </div>
                <div className="space-y-2">
                  <FieldLabel
                    label={isOllama ? "Endpoint URL (Optional)" : "API Key"}
                    help={isOllama ? "Defaults to http://localhost:11434. Use http://host.docker.internal:11434 from Docker." : "Provider API key used only for testing responses."}
                  />
                  <Input
                    type={isOllama ? "text" : "password"}
                    placeholder={
                      isOllama
                        ? "http://localhost:11434"
                        : selectedProfile?.has_api_key && !llmApiKey.trim()
                          ? "Stored on server (optional override)"
                          : "Paste provider API key"
                    }
                    value={llmApiKey}
                    onChange={(event) => setLlmApiKey(event.target.value)}
                  />
                </div>
              </div>

              <div className="flex flex-wrap gap-3">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleTestOriginal}
                  disabled={!input || !canRunLlmTest || originalTestMutation.isPending}
                  tooltip={originalTestMutation.isPending ? "Testing original prompt..." : "Test original prompt with LLM"}
                >
                  Test Original
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleTestOptimized}
                  disabled={!output || !canRunLlmTest || optimizedTestMutation.isPending}
                  tooltip={optimizedTestMutation.isPending ? "Testing optimized prompt..." : "Test optimized prompt with LLM"}
                >
                  Test Optimized
                </Button>
                <Button
                  size="sm"
                  onClick={handleTestBoth}
                  disabled={!input || !output || !canRunLlmTest || isTestPending}
                  tooltip={isTestPending ? "Running comparison test..." : "Test both prompts and compare results"}
                >
                  Test Both
                </Button>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <Card className="p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-medium text-muted-foreground">Original Prompt Response</h4>
                    <div className="flex items-center gap-2">
                      {originalTestResult && (
                        <Badge variant="outline" className="text-[10px]">
                          Provider {originalTestResult.duration_ms.toFixed(0)} ms
                        </Badge>
                      )}
                      {originalTestTiming && (
                        <Badge variant="outline" className="text-[10px]">
                          Client {originalTestTiming.wall_ms.toFixed(0)} ms
                        </Badge>
                      )}
                    </div>
                  </div>
                  <Textarea
                    readOnly
                    className="min-h-[180px] font-mono text-xs bg-muted/20"
                    placeholder="Run a test to view the original prompt response."
                    value={originalTestResult?.text ?? ""}
                  />
                </Card>
                <Card className="p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-medium text-muted-foreground">Optimized Prompt Response</h4>
                    <div className="flex items-center gap-2">
                      {optimizedTestResult && (
                        <Badge variant="outline" className="text-[10px]">
                          Provider {optimizedTestResult.duration_ms.toFixed(0)} ms
                        </Badge>
                      )}
                      {optimizedTestTiming && (
                        <Badge variant="outline" className="text-[10px]">
                          Client {optimizedTestTiming.wall_ms.toFixed(0)} ms
                        </Badge>
                      )}
                    </div>
                  </div>
                  <Textarea
                    readOnly
                    className="min-h-[180px] font-mono text-xs bg-muted/20"
                    placeholder="Run a test to view the optimized prompt response."
                    value={optimizedTestResult?.text ?? ""}
                  />
                </Card>
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}
