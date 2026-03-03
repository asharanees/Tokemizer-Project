import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { TrendingUp, PieChart, BarChart3, Activity } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Layout } from "@/components/layout/Layout";
import { authFetch } from "@/lib/authFetch";

interface StatsResponse {
  tokens_saved: number;
  cost_savings: number;
  avg_compression_percentage: number;
  avg_latency_ms: number;
  avg_quality_score: number;
  total_optimizations: number;
}

interface TelemetryPass {
  optimization_id: string;
  pass_name: string;
  pass_order: number;
  duration_ms: number;
  tokens_before: number;
  tokens_after: number;
  tokens_saved: number;
  reduction_percent: number;
  created_at: string;
}

export default function Analytics() {
  const { data: stats, isLoading: statsLoading } = useQuery<StatsResponse>({
    queryKey: ["analytics", "stats"],
    queryFn: async () => {
      const res = await authFetch("/api/v1/stats");
      if (!res.ok) throw new Error("Failed to fetch stats");
      return res.json();
    },
    refetchInterval: 10000,
  });

  const { data: telemetry, isLoading: telemetryLoading } = useQuery<TelemetryPass[]>({
    queryKey: ["analytics", "telemetry"],
    queryFn: async () => {
      const res = await authFetch("/api/v1/telemetry/recent?limit=50");
      if (!res.ok) throw new Error("Failed to fetch telemetry");
      return res.json();
    },
    refetchInterval: 15000,
  });

  const cards = [
    {
      title: "Total Optimizations",
      icon: BarChart3,
      value: stats?.total_optimizations?.toLocaleString() ?? "-",
      hint: "Prompts optimized",
    },
    {
      title: "Avg. Compression",
      icon: TrendingUp,
      value: stats ? `${stats.avg_compression_percentage.toFixed(1)}%` : "-",
      hint: "Token reduction",
    },
    {
      title: "Tokens Saved",
      icon: PieChart,
      value: stats ? stats.tokens_saved.toLocaleString() : "-",
      hint: "Total across all runs",
    },
    {
      title: "Avg Quality",
      icon: Activity,
      value: stats ? `${(stats.avg_quality_score * 100).toFixed(1)}%` : "-",
      hint: "Semantic similarity",
    },
  ];

  const telemetryRows = telemetry ?? [];
  const [page, setPage] = useState(1);
  const pageSize = 10;
  const totalPages = Math.max(1, Math.ceil(telemetryRows.length / pageSize));
  useEffect(() => {
    if (page > totalPages) {
      setPage(totalPages);
    }
  }, [page, totalPages]);
  const pagedTelemetry = telemetryRows.slice(
    (page - 1) * pageSize,
    page * pageSize
  );

  return (
    <Layout>
      <div className="space-y-6 pb-10">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold font-space-grotesk mb-2">Analytics</h1>
          <p className="text-xs sm:text-sm text-muted-foreground">Track your token optimization metrics and cost savings</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 lg:gap-6">
          {cards.map((card) => (
            <Card key={card.title} className="p-4 lg:p-6 space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-xs sm:text-sm font-medium text-muted-foreground">{card.title}</h3>
                <card.icon className="h-3.5 w-3.5 sm:h-4 sm:w-4 text-primary" />
              </div>
              {statsLoading ? (
                <Skeleton className="h-6 sm:h-7 w-16 sm:w-20" />
              ) : (
                <p className="text-lg sm:text-2xl font-bold">{card.value}</p>
              )}
              <p className="text-[10px] sm:text-xs text-muted-foreground">{card.hint}</p>
            </Card>
          ))}
        </div>

        <Card className="p-0 overflow-hidden">
          <div className="p-4 lg:p-6 border-b border-border/70">
            <h3 className="text-base lg:text-lg font-semibold">Recent Telemetry</h3>
            <p className="text-xs sm:text-sm text-muted-foreground">Per-pass timings and savings</p>
          </div>
          <div className="p-4 lg:p-6">
            {telemetryLoading ? (
              <div className="space-y-2 lg:space-y-3">
                {[1, 2, 3, 4, 5].map((i) => (
                  <Skeleton key={i} className="h-8 w-full" />
                ))}
              </div>
            ) : telemetryRows.length > 0 ? (
              <div className="space-y-4">
                <div className="rounded-md border border-border overflow-x-auto">
                  <Table className="text-xs sm:text-sm">
                    <TableHeader>
                      <TableRow>
                        <TableHead className="whitespace-nowrap">Pass</TableHead>
                        <TableHead className="whitespace-nowrap hidden sm:table-cell">Tokens</TableHead>
                        <TableHead className="whitespace-nowrap">Saved</TableHead>
                        <TableHead className="whitespace-nowrap hidden md:table-cell">Duration</TableHead>
                        <TableHead className="whitespace-nowrap hidden lg:table-cell">Run</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {pagedTelemetry.map((row) => (
                        <TableRow key={`${row.optimization_id}-${row.pass_order}`}>
                          <TableCell>
                            <div className="font-medium truncate">{row.pass_name}</div>
                            <div className="text-[9px] sm:text-xs text-muted-foreground">#{row.pass_order}</div>
                          </TableCell>
                          <TableCell className="text-[10px] sm:text-xs text-muted-foreground hidden sm:table-cell whitespace-nowrap">
                            {row.tokens_before.toLocaleString()} → {row.tokens_after.toLocaleString()}
                          </TableCell>
                          <TableCell className="text-primary font-mono text-[10px] sm:text-xs whitespace-nowrap">
                            {row.tokens_saved.toLocaleString()} ({row.reduction_percent.toFixed(1)}%)
                          </TableCell>
                          <TableCell className="text-[10px] sm:text-xs hidden md:table-cell whitespace-nowrap">{row.duration_ms.toFixed(1)} ms</TableCell>
                          <TableCell className="text-[9px] sm:text-xs text-muted-foreground hidden lg:table-cell whitespace-nowrap">
                            {new Date(row.created_at).toLocaleString()}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 sm:gap-0">
                  <p className="text-[10px] sm:text-xs text-muted-foreground">
                    Showing {(page - 1) * pageSize + 1}-{Math.min(page * pageSize, telemetryRows.length)} of {telemetryRows.length}
                  </p>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-xs"
                      onClick={() => setPage((current) => Math.max(1, current - 1))}
                      disabled={page === 1}
                      tooltip="Go to previous page of analytics"
                    >
                      Previous
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-xs"
                      onClick={() => setPage((current) => Math.min(totalPages, current + 1))}
                      disabled={page === totalPages}
                      tooltip="Go to next page of analytics"
                    >
                      Next
                    </Button>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-muted-foreground text-xs sm:text-sm">
                No telemetry available yet. Run an optimization to populate data.
              </p>
            )}
          </div>
        </Card>
      </div>
    </Layout>
  );
}
