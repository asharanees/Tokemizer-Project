import { useQuery } from "@tanstack/react-query";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Zap } from "lucide-react";
import { authFetch } from "@/lib/authFetch";

interface UsageData {
    calls_used: number;
    quota_limit: number;
    remaining: number;
    subscription_tier: string;
}

export function QuotaWidget() {
    const { data, isLoading } = useQuery<UsageData>({
        queryKey: ["usage"],
        queryFn: async () => {
            const res = await authFetch("/api/usage");
            if (!res.ok) throw new Error("Failed to fetch usage");
            return res.json();
        },
        refetchInterval: 30000, // Every 30s
    });

    if (isLoading || !data) return null;

    const percentage = Math.min(100, (data.calls_used / data.quota_limit) * 100);
    const isCritical = percentage > 80;

    return (
        <div className="space-y-3 p-3 bg-sidebar-accent/30 rounded-lg border border-sidebar-border/50">
            <div className="flex justify-between items-center text-xs">
                <span className="text-muted-foreground font-medium flex items-center gap-1.5">
                    <Zap className="w-3 h-3 text-primary" />
                    Usage
                </span>
                <Badge variant="outline" className="text-[10px] py-0 px-1.5 h-4 bg-background/50">
                    {data.subscription_tier.toUpperCase()}
                </Badge>
            </div>

            <div className="space-y-1.5">
                <div className="flex justify-between text-[10px] font-mono">
                    <span className={isCritical ? "text-destructive font-bold" : "text-muted-foreground"}>
                        {data.calls_used.toLocaleString()}
                    </span>
                    <span className="text-muted-foreground">
                        {data.quota_limit.toLocaleString()}
                    </span>
                </div>
                <Progress value={percentage} className="h-1" />
            </div>

            {isCritical && (
                <p className="text-[10px] text-destructive animate-pulse font-medium">
                    Quota nearly exhausted!
                </p>
            )}
        </div>
    );
}
