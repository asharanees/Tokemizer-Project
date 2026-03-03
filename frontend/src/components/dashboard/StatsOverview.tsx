import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { LucideIcon } from "lucide-react";
import { Zap, DollarSign, Clock, BarChart, ArrowDown, ArrowUp } from "lucide-react";
import { cn } from "@/lib/utils";

interface StatProps {
  title: string;
  value: string;
  change?: string;
  trend?: "up" | "down";
  icon: LucideIcon;
  color: "primary" | "chart-2" | "chart-3" | "chart-4";
}

function StatCard({ title, value, change, trend, icon: Icon, color }: StatProps) {
  return (
    <Card className="glass-card border-l-4" style={{ borderLeftColor: `hsl(var(--${color}))` }}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 px-4 py-3 lg:px-6 lg:py-4">
        <CardTitle className="text-xs sm:text-sm font-medium text-muted-foreground">{title}</CardTitle>
        <div className={cn("p-2 rounded-md bg-secondary/50", `text-${color}`)}>
          <Icon className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
        </div>
      </CardHeader>
      <CardContent className="px-4 py-0 lg:px-6 pb-4 lg:pb-6">
        <div className="text-xl sm:text-2xl font-bold tracking-tight font-display" data-testid={`stat-${title.toLowerCase().replace(/\s+/g, '-')}`}>
          {value}
        </div>
        {change && trend && (
          <p className="text-[10px] sm:text-xs text-muted-foreground mt-1 flex items-center gap-1">
            {trend === "down" ? (
              <ArrowDown className="h-2.5 w-2.5 sm:h-3 sm:w-3 text-green-500 shrink-0" />
            ) : (
              <ArrowUp className="h-2.5 w-2.5 sm:h-3 sm:w-3 text-green-500 shrink-0" />
            )}
            <span className="text-green-500 font-medium">{change}</span>
            <span className="opacity-70">vs last month</span>
          </p>
        )}
      </CardContent>
    </Card>
  );
}

interface StatsOverviewProps {
  stats?: {
    tokens_saved: number;
    cost_savings: number;
    avg_compression_percentage: number;
    avg_latency_ms: number;
    avg_quality_score: number;
    total_optimizations: number;
  };
}

export function StatsOverview({ stats }: StatsOverviewProps) {
  if (!stats) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 lg:gap-4">
        {[1, 2, 3, 4].map((i) => (
          <Card key={i} className="glass-card animate-pulse">
            <CardContent className="h-24" />
          </Card>
        ))}
      </div>
    );
  }

  const formatTokens = (tokens: number) => {
    if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(1)}M`;
    if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}K`;
    return tokens.toString();
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 lg:gap-4">
      <StatCard 
        title="Tokens Saved" 
        value={formatTokens(stats.tokens_saved)} 
        icon={Zap} 
        color="primary" 
      />
      <StatCard 
        title="Cost Reduction" 
        value={`$${stats.cost_savings.toFixed(2)}`}
        icon={DollarSign} 
        color="chart-2" 
      />
      <StatCard 
        title="Avg Latency" 
        value={`${stats.avg_latency_ms.toFixed(0)}ms`}
        icon={Clock} 
        color="chart-3" 
      />
      <StatCard 
        title="Quality Score" 
        value={`${(stats.avg_quality_score * 100).toFixed(1)}%`}
        icon={BarChart} 
        color="chart-4" 
      />
    </div>
  );
}
