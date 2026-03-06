import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Layers, Play, RefreshCw } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { useToast } from "@/hooks/use-toast";
import { Layout } from "@/components/layout/Layout";
import { authFetch } from "@/lib/authFetch";

interface BatchJob {
  id: string;
  name: string;
  status: string;
  total_items: number;
  processed_items: number;
  savings_percentage: number | null;
  processing_time_ms: number | null;
  created_at: string;
  completed_at: string | null;
}

interface BatchResult {
  optimized_output: string;
  stats: {
    token_savings: number;
    compression_percentage: number;
    semantic_similarity?: number | null;
  };
}

interface BatchResponse {
  batch_job_id: string;
  results: BatchResult[];
  summary: {
    total_items: number;
    avg_compression: number;
    total_processing_time_ms: number;
    throughput_prompts_per_second: number;
  };
}

interface LastBatchRun {
  jobId: string;
  summary: BatchResponse["summary"];
  entries: {
    prompt: string;
    optimized_prompt: string;
    tokens_saved: number;
    compression_percentage: number;
    semantic_similarity: number | null;
  }[];
}

export default function BatchJobs() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [promptsInput, setPromptsInput] = useState("");
  const [technique, setTechnique] = useState<"rule_based" | "llm_based">("rule_based");
  const [mode, setMode] = useState<"conservative" | "balanced" | "maximum">("balanced");
  const [batchName, setBatchName] = useState("");
  const [queryHint, setQueryHint] = useState("");
  const [segmentSpans, setSegmentSpans] = useState("");
  const [customCanonicals, setCustomCanonicals] = useState("");
  const [lastBatch, setLastBatch] = useState<LastBatchRun | null>(null);
  const sampleTemplates = [
    {
      label: "Support summary",
      prompts: [
        "Summarize this support ticket and list next steps.",
        "Rewrite this response to be concise and friendly.",
      ],
    },
    {
      label: "Product brief",
      prompts: [
        "Turn this product doc into a 5-bullet briefing.",
        "Extract key requirements and constraints.",
      ],
    },
  ];

  const { data, isLoading } = useQuery<{ batch_jobs: BatchJob[] }>({
    queryKey: ["batch-jobs", "full"],
    queryFn: async () => {
      const res = await authFetch("/api/v1/batch-jobs?limit=50");
      if (!res.ok) throw new Error("Failed to fetch batch jobs");
      return res.json();
    },
    refetchInterval: 5000,
  });

  const jobs = data?.batch_jobs ?? [];
  const summary = useMemo(() => {
    const active = jobs.filter((j) => j.status !== "completed").length;
    const completed = jobs.filter((j) => j.status === "completed").length;
    const processed = jobs.reduce((sum, j) => sum + j.processed_items, 0);
    return { active, completed, processed };
  }, [jobs]);

  const createBatch = useMutation({
    mutationFn: async (): Promise<{ response: BatchResponse; prompts: string[] }> => {
      const prompts = promptsInput
        .split("\n")
        .map((p) => p.trim())
        .filter(Boolean);
      if (!prompts.length) {
        throw new Error("Provide at least one prompt");
      }
      const payload: Record<string, unknown> = {
        prompts,
        optimization_technique: technique,
        optimization_mode: mode,
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

      const res = await authFetch("/api/v1/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Failed to create batch");
      }
      const response: BatchResponse = await res.json();
      return { response, prompts };
    },
    onSuccess: ({ response, prompts }) => {
      setPromptsInput("");
      const entries = response.results.map((result, index) => ({
        prompt: prompts[index] ?? `Prompt ${index + 1}`,
        optimized_prompt: result.optimized_output,
        tokens_saved: result.stats.token_savings,
        compression_percentage: result.stats.compression_percentage,
        semantic_similarity: result.stats.semantic_similarity ?? null,
      }));
      setLastBatch({
        jobId: response.batch_job_id,
        summary: response.summary,
        entries,
      });
      toast({ title: "Batch created", description: "Processing started." });
      queryClient.invalidateQueries({ queryKey: ["batch-jobs", "full"] });
      queryClient.invalidateQueries({ queryKey: ["batch-jobs"] });
    },
    onError: (error) => {
      toast({
        title: "Batch creation failed",
        description: error instanceof Error ? error.message : "Unable to start batch.",
        variant: "destructive",
      });
    },
  });

  const formatTime = (ms: number | null) => {
    if (!ms) return "-";
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const applyTemplate = (prompts: string[], mode: "replace" | "append") => {
    const text = prompts.join("\n");
    if (mode === "append" && promptsInput.trim()) {
      setPromptsInput((prev) => `${prev.trim()}\n${text}`);
    } else {
      setPromptsInput(text);
    }
  };

  const handleCsvUpload = (file: File | null) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const content = String(reader.result ?? "");
      const parsed = content
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => line.split(",")[0]?.replace(/^"|"$/g, "").trim())
        .filter(Boolean);
      if (!parsed.length) {
        toast({ title: "No prompts found", description: "CSV file is empty.", variant: "destructive" });
        return;
      }
      setPromptsInput((prev) => (prev.trim() ? `${prev.trim()}\n${parsed.join("\n")}` : parsed.join("\n")));
      toast({ title: "CSV loaded", description: `Added ${parsed.length} prompts.` });
    };
    reader.readAsText(file);
  };

  return (
    <Layout>
      <div className="space-y-6 pb-10">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold font-space-grotesk mb-2">Batch Jobs</h1>
            <p className="text-xs sm:text-sm text-muted-foreground">Process multiple prompts at scale with batch optimization</p>
          </div>
          <div className="flex flex-col gap-2">
            <div className="flex gap-2 flex-wrap">
              {[
                { key: "rule_based", label: "Rule based (FAST)" },
                { key: "llm_based", label: "LLM based (ADVANCED)" },
              ].map((item) => (
                <Button
                  key={item.key}
                  variant={technique === item.key ? "default" : "outline"}
                  size="sm"
                  className="text-xs sm:text-sm"
                  onClick={() => setTechnique(item.key as typeof technique)}
                >
                  {item.label}
                </Button>
              ))}
            </div>
            <div className="flex gap-2 flex-wrap">
              {["conservative", "balanced", "maximum"].map((m) => (
                <Button
                  key={m}
                  variant={mode === m ? "default" : "outline"}
                  size="sm"
                  className="text-xs sm:text-sm"
                  onClick={() => setMode(m as typeof mode)}
                  tooltip={`Set optimization mode to ${m} - ${m === "conservative" ? "Minimal changes, maximum safety" : m === "balanced" ? "Moderate optimization with good safety" : "Maximum optimization, higher risk"}`}
                >
                  {m}
                </Button>
              ))}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 lg:gap-6">
          <Card className="p-4 lg:p-6 space-y-2">
            <p className="text-xs sm:text-sm font-medium text-muted-foreground">Active Jobs</p>
            {isLoading ? <Skeleton className="h-6 w-10" /> : <p className="text-xl sm:text-2xl font-bold">{summary.active}</p>}
          </Card>
          <Card className="p-4 lg:p-6 space-y-2">
            <p className="text-xs sm:text-sm font-medium text-muted-foreground">Completed</p>
            {isLoading ? <Skeleton className="h-6 w-10" /> : <p className="text-xl sm:text-2xl font-bold">{summary.completed}</p>}
          </Card>
          <Card className="p-4 lg:p-6 space-y-2">
            <p className="text-xs sm:text-sm font-medium text-muted-foreground">Total Processed</p>
            {isLoading ? (
              <Skeleton className="h-6 w-16" />
            ) : (
              <p className="text-xl sm:text-2xl font-bold">{summary.processed.toLocaleString()}</p>
            )}
          </Card>
        </div>

        <Card className="p-4 lg:p-6 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
            <div>
              <p className="text-sm font-medium">Create Batch Job</p>
              <p className="text-xs text-muted-foreground">One prompt per line, processed synchronously.</p>
            </div>
            <Button
              size="sm"
              className="gap-2 w-full sm:w-auto text-xs sm:text-sm"
              onClick={() => createBatch.mutate()}
              disabled={createBatch.isPending || !promptsInput.trim()}
              tooltip={createBatch.isPending ? "Creating batch job..." : "Start processing batch of prompts"}
            >
              {createBatch.isPending ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {createBatch.isPending ? "Starting..." : "Start Batch"}
            </Button>
          </div>
          <Textarea
            value={promptsInput}
            onChange={(e) => setPromptsInput(e.target.value)}
            placeholder="Prompt 1&#10;Prompt 2&#10;Prompt 3"
            className="min-h-[160px] text-xs sm:text-sm"
          />
          <Accordion type="single" collapsible>
            <AccordionItem value="additional-controls">
              <AccordionTrigger className="text-sm font-medium">Additional Controls (Optional)</AccordionTrigger>
              <AccordionContent>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <Card className="p-4 space-y-3">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Batch Name</p>
                    <Input
                      placeholder="e.g. support-q1"
                      value={batchName}
                      onChange={(event) => setBatchName(event.target.value)}
                    />
                    <p className="text-xs text-muted-foreground">
                      Optional label shown in batch dashboards.
                    </p>
                  </Card>

                  <Card className="p-4 space-y-3">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Query Hint (RAG only)</p>
                    <Input
                      placeholder="What should be prioritized in compression?"
                      value={queryHint}
                      onChange={(event) => setQueryHint(event.target.value)}
                    />
                  </Card>

                  <Card className="p-4 space-y-3">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Segment Spans</p>
                    <Textarea
                      placeholder='[{"start": 0, "end": 42, "weight": 1.0}]'
                      value={segmentSpans}
                      onChange={(event) => setSegmentSpans(event.target.value)}
                      className="min-h-[120px] font-mono text-xs"
                    />
                  </Card>

                  <Card className="p-4 space-y-3">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Custom Canonicals</p>
                    <Textarea
                      placeholder='{"Natural Language Processing":"NLP","Machine Learning":"ML"}'
                      value={customCanonicals}
                      onChange={(event) => setCustomCanonicals(event.target.value)}
                      className="min-h-[120px] font-mono text-xs"
                    />
                  </Card>
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap gap-2">
              {sampleTemplates.map((template) => (
                <div key={template.label} className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => applyTemplate(template.prompts, "replace")}
                    tooltip={`Replace current prompts with ${template.label} template`}
                  >
                    Use {template.label}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => applyTemplate(template.prompts, "append")}
                    tooltip={`Append ${template.label} template to current prompts`}
                  >
                    Append
                  </Button>
                </div>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Upload CSV</span>
              <Input
                type="file"
                accept=".csv,text/csv"
                onChange={(event) => handleCsvUpload(event.target.files?.[0] ?? null)}
                className="h-8 w-[220px]"
              />
            </div>
          </div>
        </Card>

        {lastBatch && (
          <Card className="p-6 space-y-4">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div>
                <p className="text-sm font-medium">Latest Batch Results</p>
                <p className="text-xs text-muted-foreground">
                  Job {lastBatch.jobId} - {lastBatch.summary.total_items} prompts processed
                </p>
              </div>
              <div className="flex gap-2">
                <Badge variant="secondary">Avg Compression {lastBatch.summary.avg_compression.toFixed(1)}%</Badge>
                <Badge variant="secondary">Latency {formatTime(lastBatch.summary.total_processing_time_ms)}</Badge>
                <Badge variant="secondary">
                  Throughput {lastBatch.summary.throughput_prompts_per_second.toFixed(2)} req/s
                </Badge>
              </div>
            </div>

            <div className="rounded-md border border-border overflow-auto max-h-[400px]">
              <Table>
                <TableHeader className="sticky top-0 bg-secondary/80 backdrop-blur-sm z-10 shadow-sm">
                  <TableRow>
                    <TableHead className="w-12">#</TableHead>
                    <TableHead>Prompt</TableHead>
                    <TableHead>Optimized</TableHead>
                    <TableHead>Saved</TableHead>
                    <TableHead>Similarity</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {lastBatch.entries.map((entry, index) => (
                    <TableRow key={`${lastBatch.jobId}-${index}`}>
                      <TableCell className="text-muted-foreground font-mono">{index + 1}</TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        <div className="line-clamp-2">{entry.prompt}</div>
                      </TableCell>
                      <TableCell className="text-sm">
                        <div className="line-clamp-2 text-foreground">{entry.optimized_prompt}</div>
                      </TableCell>
                      <TableCell className="font-mono text-primary">
                        {entry.tokens_saved.toLocaleString()} ({entry.compression_percentage.toFixed(1)}%)
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {entry.semantic_similarity !== null ? `${(entry.semantic_similarity * 100).toFixed(1)}%` : "-"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </Card>
        )}

        <Card className="p-0 overflow-hidden">
          <div className="p-4 border-b border-border flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Recent Batches</p>
              <p className="text-xs text-muted-foreground">Auto-refreshing</p>
            </div>
            <Badge variant="secondary">Live</Badge>
          </div>
          <div className="p-4">
            {isLoading ? (
              <div className="space-y-3">
                {[1, 2, 3, 4].map((i) => (
                  <Skeleton key={i} className="h-9 w-full" />
                ))}
              </div>
            ) : jobs.length === 0 ? (
              <div className="p-8 text-center space-y-4 border border-dashed border-border rounded-md">
                <Layers className="h-10 w-10 text-muted-foreground mx-auto opacity-50" />
                <p className="text-sm text-muted-foreground">No batch jobs yet. Add prompts above to start one.</p>
              </div>
            ) : (
              <div className="rounded-md border border-border overflow-auto max-h-[500px]">
                <Table>
                  <TableHeader className="sticky top-0 bg-secondary/80 backdrop-blur-sm z-10 shadow-sm">
                    <TableRow>
                      <TableHead>Job</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Progress</TableHead>
                      <TableHead>Savings</TableHead>
                      <TableHead>Duration</TableHead>
                      <TableHead>Created</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {jobs.map((job) => {
                      const progress =
                        job.total_items > 0 ? Math.floor((job.processed_items / job.total_items) * 100) : 0;
                      return (
                        <TableRow key={job.id}>
                          <TableCell className="font-medium">{job.name}</TableCell>
                          <TableCell>
                            {job.status === "completed" && (
                              <Badge variant="default" className="bg-green-500/15 text-green-500">
                                Completed
                              </Badge>
                            )}
                            {job.status === "processing" && (
                              <Badge variant="secondary" className="animate-pulse">
                                Processing
                              </Badge>
                            )}
                            {job.status === "failed" && <Badge variant="destructive">Failed</Badge>}
                            {job.status === "pending" && <Badge variant="outline">Pending</Badge>}
                          </TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {job.processed_items}/{job.total_items} ({progress}%)
                          </TableCell>
                          <TableCell className="font-mono text-primary">
                            {job.savings_percentage ? `${job.savings_percentage.toFixed(1)}%` : "-"}
                          </TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {formatTime(job.processing_time_ms)}
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground">
                            {new Date(job.created_at).toLocaleString()}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            )}
          </div>
        </Card>
      </div>
    </Layout>
  );
}
