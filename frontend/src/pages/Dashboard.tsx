import { Layout } from "@/components/layout/Layout";
import { StatsOverview } from "@/components/dashboard/StatsOverview";
import { RecentBatches } from "@/components/dashboard/RecentBatches";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Plus, UploadCloud, Lightbulb, ArrowRight } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { NavigationBreadcrumb } from "@/components/layout/NavigationBreadcrumb";
import { HelpTooltip } from "@/components/ui/HelpTooltip";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import { ToastAction } from "@/components/ui/toast";
import { RefreshCw } from "lucide-react";
import { authFetch } from "@/lib/authFetch";

const chartData = [
  { name: "Mon", original: 4000, optimized: 2400 },
  { name: "Tue", original: 3000, optimized: 1398 },
  { name: "Wed", original: 2000, optimized: 980 },
  { name: "Thu", original: 2780, optimized: 1908 },
  { name: "Fri", original: 1890, optimized: 1000 },
  { name: "Sat", original: 2390, optimized: 1400 },
  { name: "Sun", original: 3490, optimized: 2100 },
];

export default function Dashboard() {
  const { user } = useAuth();
  const { toast } = useToast();
  const { data: stats, refetch, isFetching } = useQuery({
    queryKey: ["stats"],
    queryFn: async () => {
      const response = await authFetch("/api/v1/stats");
      if (!response.ok) throw new Error("Failed to fetch stats");
      return response.json();
    },
    refetchInterval: 10000,
  });

  const handleRefresh = async () => {
    await refetch();
    toast({
      title: "Stats Refreshed",
      description: "Data has been updated to the latest available metrics.",
      action: (
        <ToastAction altText="View Analytics" onClick={() => window.location.href = "/analytics"}>
          Analyze
        </ToastAction>
      )
    });
  };

  const trendData = Array.isArray(stats?.trend) ? stats.trend : chartData;
  const showingSample = !Array.isArray(stats?.trend);

  return (
    <Layout>
      <div className="space-y-6 lg:space-y-8">
        <NavigationBreadcrumb />
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight font-display text-glow" data-testid="heading-dashboard">
              Welcome back, {user?.name?.split(' ')[0] || "there"}!
            </h1>
            <p className="text-xs sm:text-sm text-muted-foreground mt-1">
              Here's what's happening with your token optimizations.
              <HelpTooltip content="Stats are aggregated monthly and updated every 10 seconds." />
            </p>
          </div>
          <div className="flex flex-col sm:flex-row gap-2 w-full sm:w-auto">
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={isFetching}
              className="rounded-full w-full sm:w-auto"
              tooltip="Refresh dashboard statistics"
            >
              <RefreshCw className={`w-4 h-4 ${isFetching ? 'animate-spin' : ''}`} />
            </Button>
            <Button variant="outline" size="sm" className="gap-2 w-full sm:w-auto text-xs sm:text-sm" data-testid="button-upload-batch" asChild tooltip="Upload multiple prompts for batch processing">
              <Link href="/batch">
                <UploadCloud className="w-4 h-4" />
                <span className="hidden sm:inline">Upload Batch</span>
                <span className="sm:hidden">Batch</span>
              </Link>
            </Button>
            <Button size="sm" className="gap-2 shadow-lg shadow-primary/20 w-full sm:w-auto text-xs sm:text-sm" data-testid="button-new-optimization" asChild tooltip="Start a new prompt optimization">
              <Link href="/playground">
                <Plus className="w-4 h-4" />
                <span className="hidden sm:inline">New Optimization</span>
                <span className="sm:hidden">New</span>
              </Link>
            </Button>
          </div>
        </div>

        <StatsOverview stats={stats} />

        <div className="grid grid-cols-1 lg:grid-cols-7 gap-4 lg:gap-6">
          <Card className="lg:col-span-4 glass-card border-none">
            <CardHeader className="flex flex-row items-center justify-between pb-3 lg:pb-4">
              <CardTitle className="text-lg lg:text-base">Token Consumption Trend</CardTitle>
              {showingSample && <Badge variant="outline" className="text-xs">Sample data</Badge>}
            </CardHeader>
            <CardContent>
              {showingSample && (
                <p className="text-xs text-muted-foreground mb-3">
                  Historical trends will appear after optimizations are recorded.
                </p>
              )}
              <div className="h-[250px] sm:h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={250}>
                  <AreaChart data={trendData}>
                    <defs>
                      <linearGradient id="colorOriginal" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--muted-foreground))" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="hsl(var(--muted-foreground))" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="colorOptimized" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                    <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => `${value / 1000}k`} />
                    <Tooltip
                      contentStyle={{ backgroundColor: "hsl(var(--card))", borderColor: "hsl(var(--border))" }}
                      itemStyle={{ color: "hsl(var(--foreground))" }}
                    />
                    <Area type="monotone" dataKey="original" stroke="hsl(var(--muted-foreground))" fillOpacity={1} fill="url(#colorOriginal)" strokeWidth={2} />
                    <Area type="monotone" dataKey="optimized" stroke="hsl(var(--primary))" fillOpacity={1} fill="url(#colorOptimized)" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card className="lg:col-span-3 glass-card border-none">
            <CardHeader>
              <CardTitle className="text-lg lg:text-base">Savings by Algorithm</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 lg:space-y-5">
                {[
                  { name: "Stop Word Removal", value: 35, color: "bg-primary" },
                  { name: "Entity Canonicalization", value: 28, color: "bg-chart-2" },
                  { name: "Whitespace Normalization", value: 15, color: "bg-chart-3" },
                  { name: "Semantic Deduplication", value: 12, color: "bg-chart-4" },
                  { name: "Other", value: 10, color: "bg-chart-5" }
                ].map((item) => (
                  <div key={item.name} className="space-y-1">
                    <div className="flex items-center justify-between text-xs sm:text-sm">
                      <span className="truncate pr-2">{item.name}</span>
                      <span className="font-mono shrink-0">{item.value}%</span>
                    </div>
                    <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                      <div className={`h-full ${item.color}`} style={{ width: `${item.value}%` }}></div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 lg:gap-6">
          <div className="lg:col-span-2">
            <RecentBatches />
          </div>
          <Card className="glass-card border-none h-fit">
            <CardHeader className="flex flex-row items-center gap-2 pb-3 lg:pb-4">
              <Lightbulb className="w-5 h-5 text-yellow-500 shrink-0" />
              <CardTitle className="text-base lg:text-lg">Pro Tips</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="p-3 bg-primary/5 rounded-lg border border-primary/10">
                <p className="text-xs font-semibold text-primary mb-1 italic">Did you know?</p>
                <p className="text-xs text-muted-foreground">
                  Using "Canonical Mappings" can reduce token count by up to 15% by normalizing frequent entity names.
                </p>
              </div>
              <Link href="/canonical">
                <Button variant="link" className="p-0 h-auto text-xs gap-1">
                  Configure Mappings <ArrowRight className="w-3 h-3" />
                </Button>
              </Link>
              <Separator />
              <div className="space-y-2">
                <p className="text-xs font-medium">Optimization Checklist:</p>
                <ul className="text-[10px] sm:text-[11px] text-muted-foreground space-y-1">
                  <li className="flex gap-2">
                    <div className="w-1 h-1 rounded-full bg-primary mt-1.5 shrink-0" />
                    <span>Review cost-benefit of semantic deduplication</span>
                  </li>
                  <li className="flex gap-2">
                    <div className="w-1 h-1 rounded-full bg-primary mt-1.5 shrink-0" />
                    <span>Validate regex patterns in unit normalizer</span>
                  </li>
                  <li className="flex gap-2">
                    <div className="w-1 h-1 rounded-full bg-primary mt-1.5 shrink-0" />
                    <span>Set up automated monthly usage reports</span>
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
}
