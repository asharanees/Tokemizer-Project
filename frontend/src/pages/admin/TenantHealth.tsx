import { useQuery } from "@tanstack/react-query";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { authFetch } from "@/lib/authFetch";

interface TenantHealthRow {
  tenant_id: string;
  email: string;
  subscription_tier: string;
  error_rate_pct: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  token_throughput: number;
  quota_limit: number;
  quota_used: number;
  quota_burndown_pct: number;
}

export default function TenantHealth() {
  const { data, isLoading } = useQuery<{ tenants: TenantHealthRow[] }>({
    queryKey: ["admin-tenant-health"],
    queryFn: async () => {
      const res = await authFetch("/api/admin/tenant-health");
      if (!res.ok) throw new Error("Failed to fetch tenant health");
      return res.json();
    },
    refetchInterval: 10000,
  });

  return (
    <Layout>
      <div className="space-y-6">
        <h1 className="text-3xl font-bold tracking-tight">Tenant Health Dashboard</h1>
        <Card className="glass-card">
          <CardHeader>
            <CardTitle>Per-Tenant Reliability & Quota Burn-down</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Tenant</TableHead>
                  <TableHead>Error Rate</TableHead>
                  <TableHead>p95 / p99 Latency (ms)</TableHead>
                  <TableHead>Token Throughput</TableHead>
                  <TableHead>Quota Burn-down</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  <TableRow><TableCell colSpan={5}>Loading...</TableCell></TableRow>
                ) : (
                  (data?.tenants || []).map((row) => (
                    <TableRow key={row.tenant_id}>
                      <TableCell>
                        <div className="text-xs">
                          <div className="font-medium">{row.email}</div>
                          <div className="text-muted-foreground">{row.subscription_tier}</div>
                        </div>
                      </TableCell>
                      <TableCell>{row.error_rate_pct.toFixed(2)}%</TableCell>
                      <TableCell>{row.p95_latency_ms.toFixed(2)} / {row.p99_latency_ms.toFixed(2)}</TableCell>
                      <TableCell>{row.token_throughput.toLocaleString()}</TableCell>
                      <TableCell>{row.quota_used.toLocaleString()} / {row.quota_limit.toLocaleString()} ({row.quota_burndown_pct.toFixed(1)}%)</TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}
