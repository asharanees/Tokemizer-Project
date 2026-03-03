import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { MoreHorizontal, FileText } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
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

export function RecentBatches() {
  const [, navigate] = useLocation();

  const { data: batchData } = useQuery({
    queryKey: ["batch-jobs"],
    queryFn: async () => {
      const response = await authFetch("/api/v1/batch-jobs?limit=5");
      if (!response.ok) throw new Error("Failed to fetch batch jobs");
      return response.json() as Promise<{ batch_jobs: BatchJob[] }>;
    },
    refetchInterval: 5000,
  });

  const jobs = batchData?.batch_jobs || [];

  const formatTime = (ms: number | null) => {
    if (!ms) return "N/A";
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  return (
    <div className="space-y-3 lg:space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 sm:gap-0">
        <h3 className="text-base lg:text-lg font-display font-medium">Recent Batch Jobs</h3>
        <Button variant="outline" size="sm" className="text-xs sm:text-sm w-full sm:w-auto" data-testid="button-view-all-batches" onClick={() => navigate("/batch")} tooltip="View all batch jobs">
          View All
        </Button>
      </div>
      <div className="rounded-md border border-border bg-card/50 overflow-x-auto">
        {jobs.length === 0 ? (
          <div className="p-6 lg:p-8 text-center text-muted-foreground text-xs sm:text-sm">
            No batch jobs yet. Upload a batch to get started.
          </div>
        ) : (
          <Table className="text-xs sm:text-sm">
            <TableHeader>
              <TableRow className="hover:bg-transparent border-border">
                <TableHead className="whitespace-nowrap w-[180px] sm:w-[250px]">Job Name</TableHead>
                <TableHead className="whitespace-nowrap">Status</TableHead>
                <TableHead className="whitespace-nowrap hidden sm:table-cell">Items</TableHead>
                <TableHead className="whitespace-nowrap hidden md:table-cell">Savings</TableHead>
                <TableHead className="whitespace-nowrap hidden lg:table-cell">Duration</TableHead>
                <TableHead className="text-right whitespace-nowrap">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.map((job) => (
                <TableRow key={job.id} className="hover:bg-muted/50 border-border" data-testid={`row-batch-${job.id}`}>
                  <TableCell className="font-medium">
                    <div className="flex items-center gap-1 sm:gap-2">
                      <FileText className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-muted-foreground shrink-0" />
                      <span className="truncate">{job.name}</span>
                    </div>
                    <div className="text-[9px] sm:text-xs text-muted-foreground pl-4 sm:pl-6 truncate">job_{job.id}</div>
                  </TableCell>
                  <TableCell>
                    {job.status === "completed" && (
                      <Badge variant="default" className="bg-green-500/15 text-green-400 hover:bg-green-500/25 border-transparent text-[10px] sm:text-xs">Completed</Badge>
                    )}
                    {job.status === "processing" && (
                      <Badge variant="secondary" className="animate-pulse bg-primary/15 text-primary hover:bg-primary/25 border-transparent text-[10px] sm:text-xs">Processing</Badge>
                    )}
                    {job.status === "failed" && (
                      <Badge variant="destructive" className="bg-red-500/15 text-red-400 hover:bg-red-500/25 border-transparent text-[10px] sm:text-xs">Failed</Badge>
                    )}
                    {job.status === "pending" && (
                      <Badge variant="secondary" className="bg-muted text-muted-foreground text-[10px] sm:text-xs">Pending</Badge>
                    )}
                  </TableCell>
                  <TableCell className="hidden sm:table-cell whitespace-nowrap">{job.total_items.toLocaleString()}</TableCell>
                  <TableCell className="font-mono text-primary hidden md:table-cell whitespace-nowrap">
                    {job.savings_percentage ? `${job.savings_percentage.toFixed(1)}%` : '~38%'}
                  </TableCell>
                  <TableCell className="text-muted-foreground">{formatTime(job.processing_time_ms)}</TableCell>
                  <TableCell className="text-right">
                    <Button variant="ghost" size="icon" data-testid={`button-batch-actions-${job.id}`} tooltip="View batch job actions">
                      <MoreHorizontal className="w-4 h-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>
    </div>
  );
}
