import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Clock } from "lucide-react";
import { Layout } from "@/components/layout/Layout";
import { authFetch } from "@/lib/authFetch";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";

interface HistoryRow {
  id: string;
  original_tokens: number;
  optimized_tokens: number;
  tokens_saved: number;
  compression_percentage: number;
  semantic_similarity: number | null;
  mode: string;
  created_at: string;
}

interface HistoryDetail {
  id: string;
  mode: string;
  created_at: string;
  updated_at: string;
  raw_prompt: string;
  optimized_prompt: string;
  original_tokens: number;
  optimized_tokens: number;
  tokens_saved: number;
  compression_percentage: number;
  semantic_similarity: number | null;
  processing_time_ms: number;
  estimated_cost_before: number;
  estimated_cost_after: number;
  estimated_cost_saved: number;
  techniques_applied: string[];
}

export default function History() {
  const { data, isLoading, isError } = useQuery<{ optimizations: HistoryRow[] }>({
    queryKey: ["history"],
    queryFn: async () => {
      const res = await authFetch("/api/v1/history?limit=200");
      if (!res.ok) throw new Error("Failed to load history");
      return res.json();
    },
    refetchInterval: 15000,
  });

  const toNumber = (value: unknown, fallback = 0) => {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };

  const toText = (value: unknown) => {
    if (value === null || value === undefined) return "";
    return String(value);
  };

  const [selectedOptimizationId, setSelectedOptimizationId] = useState<string | null>(null);
  const [isDetailOpen, setIsDetailOpen] = useState(false);

  const detailQuery = useQuery<HistoryDetail>({
    queryKey: ["history-detail", selectedOptimizationId],
    queryFn: async () => {
      if (!selectedOptimizationId) {
        throw new Error("Missing optimization id");
      }
      const res = await authFetch(`/api/v1/history/${selectedOptimizationId}`);
      if (!res.ok) throw new Error("Failed to load optimization detail");
      return res.json();
    },
    enabled: isDetailOpen && !!selectedOptimizationId,
  });

  const optimizations = data?.optimizations ?? [];
  const summary = useMemo(() => {
    if (!optimizations.length) {
      return { total: 0, avgCompression: 0, avgSimilarity: 0 };
    }
    const total = optimizations.length;
    const avgCompression =
      optimizations.reduce((sum, item) => sum + toNumber(item?.compression_percentage), 0) / total;
    const similarities = optimizations
      .map((o) => toNumber(o?.semantic_similarity, 0))
      .filter((s) => s > 0);
    const avgSimilarity = similarities.length
      ? similarities.reduce((a, b) => a + b, 0) / similarities.length
      : 0;
    return { total, avgCompression, avgSimilarity };
  }, [optimizations]);

  const [page, setPage] = useState(1);
  const pageSize = 10;
  const totalPages = Math.max(1, Math.ceil(optimizations.length / pageSize));
  useEffect(() => {
    if (page > totalPages) {
      setPage(totalPages);
    }
  }, [page, totalPages]);
  const pagedOptimizations = optimizations.slice(
    (page - 1) * pageSize,
    page * pageSize
  );
  const shouldRenderDetailContent = detailQuery.isFetching || (isDetailOpen && !!selectedOptimizationId);

  return (
    <Layout>
      <div className="space-y-6 pb-10">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold font-space-grotesk mb-2">Optimization History</h1>
          <p className="text-xs sm:text-sm text-muted-foreground">View all your past prompt optimizations</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 lg:gap-4">
          <Card className="p-4 lg:p-6 space-y-2">
            <p className="text-xs sm:text-sm text-muted-foreground">Total Optimizations</p>
            {isLoading ? <Skeleton className="h-6 w-12" /> : <p className="text-xl sm:text-2xl font-bold">{summary.total}</p>}
          </Card>
          <Card className="p-4 lg:p-6 space-y-2">
            <p className="text-xs sm:text-sm text-muted-foreground">Average Compression</p>
            {isLoading ? (
              <Skeleton className="h-6 w-16" />
            ) : (
              <p className="text-xl sm:text-2xl font-bold">{summary.avgCompression.toFixed(1)}%</p>
            )}
          </Card>
          <Card className="p-4 lg:p-6 space-y-2">
            <p className="text-xs sm:text-sm text-muted-foreground">Avg Semantic Similarity</p>
            {isLoading ? (
              <Skeleton className="h-6 w-16" />
            ) : (
              <p className="text-xl sm:text-2xl font-bold">{(summary.avgSimilarity * 100).toFixed(1)}%</p>
            )}
          </Card>
        </div>

        <Card className="p-0 overflow-hidden">
          <div className="p-4 lg:p-6 border-b border-border flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 sm:gap-0">
            <div>
              <p className="text-sm font-medium">Recent Runs</p>
              <p className="text-xs text-muted-foreground">Page {page} of {totalPages}</p>
            </div>
            <Badge variant="outline" className="font-mono w-fit">
              Live
            </Badge>
          </div>
          <div className="p-4 lg:p-6">
            {isLoading ? (
              <div className="space-y-2 lg:space-y-3">
                {[1, 2, 3, 4, 5].map((i) => (
                  <Skeleton key={i} className="h-9 w-full" />
                ))}
              </div>
            ) : optimizations.length === 0 ? (
              <div className="p-8 text-center space-y-4 border border-dashed border-border rounded-md">
                <Clock className="h-10 w-10 text-muted-foreground mx-auto opacity-50" />
                <p className="text-xs sm:text-sm text-muted-foreground">
                  {isError ? "Failed to load optimization history." : "Your optimization history will appear here."}
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="rounded-md border border-border overflow-x-auto">
                  <Table className="text-xs sm:text-sm">
                    <TableHeader>
                      <TableRow>
                        <TableHead className="whitespace-nowrap">ID</TableHead>
                        <TableHead className="whitespace-nowrap">Mode</TableHead>
                        <TableHead className="whitespace-nowrap hidden sm:table-cell">Tokens</TableHead>
                        <TableHead className="whitespace-nowrap">Compression</TableHead>
                        <TableHead className="whitespace-nowrap hidden md:table-cell">Similarity</TableHead>
                        <TableHead className="whitespace-nowrap hidden lg:table-cell">Created</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {pagedOptimizations.map((item) => (
                        <TableRow key={toText(item.id)}>
                          <TableCell className="font-mono text-[10px] sm:text-xs">
                            <Button
                              variant="link"
                              size="sm"
                              className="p-0 min-h-0 h-auto font-mono text-muted-foreground"
                              title={toText(item.id)}
                              onClick={() => {
                                setSelectedOptimizationId(toText(item.id));
                                setIsDetailOpen(true);
                              }}
                            >
                              {toText(item.id).slice(0, 8)}
                            </Button>
                          </TableCell>
                          <TableCell>
                            <Badge variant="secondary" className="capitalize text-[10px] sm:text-xs">
                              {toText(item.mode)}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-[10px] sm:text-xs text-muted-foreground hidden sm:table-cell whitespace-nowrap">
                            {toNumber(item.original_tokens).toLocaleString()} → {toNumber(item.optimized_tokens).toLocaleString()}
                          </TableCell>
                          <TableCell className="font-mono text-primary text-[10px] sm:text-xs whitespace-nowrap">
                            {toNumber(item.tokens_saved).toLocaleString()} ({toNumber(item.compression_percentage).toFixed(1)}%)
                          </TableCell>
                          <TableCell className="text-[10px] sm:text-xs hidden md:table-cell">
                            {(toNumber(item.semantic_similarity, 0) * 100).toFixed(1)}%
                          </TableCell>
                          <TableCell className="text-[9px] sm:text-xs text-muted-foreground hidden lg:table-cell whitespace-nowrap">
                            {new Date(toText(item.created_at)).toLocaleString()}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 sm:gap-0">
                  <p className="text-[10px] sm:text-xs text-muted-foreground">
                    Showing {(page - 1) * pageSize + 1}-{Math.min(page * pageSize, optimizations.length)} of {optimizations.length}
                  </p>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-xs"
                      onClick={() => setPage((current) => Math.max(1, current - 1))}
                      disabled={page === 1}
                      tooltip="Go to previous page of history"
                    >
                      Previous
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-xs"
                      onClick={() => setPage((current) => Math.min(totalPages, current + 1))}
                      disabled={page === totalPages}
                      tooltip="Go to next page of history"
                    >
                      Next
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </Card>
      </div>

      <Dialog
        open={isDetailOpen}
        onOpenChange={(open) => {
          setIsDetailOpen(open);
          if (!open) {
            setSelectedOptimizationId(null);
          }
        }}
      >
        <DialogContent className="max-w-5xl">
          <DialogHeader>
            <DialogTitle>Optimization Detail</DialogTitle>
          </DialogHeader>

          {shouldRenderDetailContent ? (
            detailQuery.isFetching ? (
              <div className="space-y-3">
                <Skeleton className="h-5 w-2/3" />
                <Skeleton className="h-5 w-1/2" />
                <Skeleton className="h-64 w-full" />
              </div>
            ) : detailQuery.isError ? (
              <div className="text-sm text-muted-foreground">Failed to load optimization detail.</div>
            ) : !detailQuery.data ? (
              <div className="text-sm text-muted-foreground">No optimization detail loaded.</div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  <div className="space-y-1">
                    <div className="text-muted-foreground">Prompt ID</div>
                    <div className="font-mono break-all">{toText(detailQuery.data.id)}</div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-muted-foreground">Created</div>
                    <div className="whitespace-nowrap">{new Date(toText(detailQuery.data.created_at)).toLocaleString()}</div>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  <Card className="p-3">
                    <div className="text-[10px] text-muted-foreground">Tokens</div>
                    <div className="font-mono text-xs">{toNumber(detailQuery.data.original_tokens).toLocaleString()} → {toNumber(detailQuery.data.optimized_tokens).toLocaleString()}</div>
                  </Card>
                  <Card className="p-3">
                    <div className="text-[10px] text-muted-foreground">Saved</div>
                    <div className="font-mono text-xs">{toNumber(detailQuery.data.tokens_saved).toLocaleString()} ({toNumber(detailQuery.data.compression_percentage).toFixed(1)}%)</div>
                  </Card>
                  <Card className="p-3">
                    <div className="text-[10px] text-muted-foreground">Similarity</div>
                    <div className="font-mono text-xs">{(toNumber(detailQuery.data.semantic_similarity, 0) * 100).toFixed(1)}%</div>
                  </Card>
                  <Card className="p-3">
                    <div className="text-[10px] text-muted-foreground">Est. Cost Saved</div>
                    <div className="font-mono text-xs">${toNumber(detailQuery.data.estimated_cost_saved, 0).toFixed(4)}</div>
                  </Card>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Original Prompt</div>
                    <ScrollArea className="h-[280px] rounded-md border border-border">
                      <pre className="p-3 text-xs whitespace-pre-wrap break-words">{toText(detailQuery.data.raw_prompt)}</pre>
                    </ScrollArea>
                  </div>
                  <div className="space-y-2">
                    <div className="text-xs font-medium">Optimized Prompt</div>
                    <ScrollArea className="h-[280px] rounded-md border border-border">
                      <pre className="p-3 text-xs whitespace-pre-wrap break-words">{toText(detailQuery.data.optimized_prompt)}</pre>
                    </ScrollArea>
                  </div>
                </div>
              </div>
            )
          ) : null}
        </DialogContent>
      </Dialog>
    </Layout>
  );
}
