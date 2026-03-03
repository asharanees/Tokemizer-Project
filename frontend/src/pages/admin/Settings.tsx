import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useEffect, useState } from "react";
import { authFetch } from "@/lib/authFetch";

interface AdminSettings {
    smtp_host: string;
    smtp_port: number;
    smtp_user: string;
    smtp_password_set?: boolean;
    stripe_secret_key_set?: boolean;
    stripe_publishable_key: string;
    log_level?: string;
    telemetry_enabled?: boolean;
    access_token_expire_minutes?: number;
    refresh_token_expire_days?: number;
    history_enabled?: boolean;
    optimizer_prewarm_models?: boolean;
    cors_origins?: string;
    smtp_from_email?: string;
}

type AdminSettingsUpdate = AdminSettings & {
    stripe_secret_key?: string;
    smtp_password?: string;
};

const parseIntegerOrFallback = (value: string, fallback: number): number => {
    const parsed = Number.parseInt(value, 10);
    return Number.isNaN(parsed) ? fallback : parsed;
};

const buildAdminSettingsUpdatePayload = (
    form: AdminSettings,
    stripeSecretKey: string,
    smtpPassword: string,
): AdminSettingsUpdate => {
    const payload: AdminSettingsUpdate = {
        smtp_host: form.smtp_host,
        smtp_port: form.smtp_port,
        smtp_user: form.smtp_user,
        stripe_publishable_key: form.stripe_publishable_key,
        log_level: form.log_level,
        telemetry_enabled: form.telemetry_enabled,
        access_token_expire_minutes: form.access_token_expire_minutes,
        refresh_token_expire_days: form.refresh_token_expire_days,
        history_enabled: form.history_enabled,
        optimizer_prewarm_models: form.optimizer_prewarm_models,
        cors_origins: form.cors_origins,
        smtp_from_email: form.smtp_from_email,
    };

    if (stripeSecretKey) {
        payload.stripe_secret_key = stripeSecretKey;
    }
    if (smtpPassword) {
        payload.smtp_password = smtpPassword;
    }

    return payload;
};

export default function AdminSettingsPage() {
    const { toast } = useToast();
    const queryClient = useQueryClient();
    const [form, setForm] = useState<AdminSettings | null>(null);
    const [stripeSecretKey, setStripeSecretKey] = useState<string>("");
    const [smtpPassword, setSmtpPassword] = useState<string>("");

    const { data, isLoading } = useQuery<AdminSettings>({
        queryKey: ["admin-settings"],
        queryFn: async () => {
            const res = await authFetch("/api/admin/settings");
            if (!res.ok) throw new Error("Failed to fetch settings");
            return res.json();
        },
    });

    useEffect(() => {
        if (data) setForm(data);
    }, [data]);

    const updateMutation = useMutation({
        mutationFn: async (updated: AdminSettingsUpdate) => {
            const res = await authFetch("/api/admin/settings", {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(updated),
            });
            if (!res.ok) {
                let detail = "Failed to update settings";
                try {
                    const payload = await res.json();
                    if (payload?.detail && typeof payload.detail === "string") {
                        detail = payload.detail;
                    }
                } catch {
                    // Keep generic message when response body is not JSON.
                }
                throw new Error(detail);
            }
            return res.json();
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["admin-settings"] });
            setStripeSecretKey(""); // Clear the input after save
            setSmtpPassword("");
            toast({ title: "Settings updated successfully" });
        },
        onError: (error: Error) => {
            toast({ title: "Error", description: error.message, variant: "destructive" });
        },
    });

    const handleSave = () => {
        if (form) {
            const toUpdate = buildAdminSettingsUpdatePayload(form, stripeSecretKey, smtpPassword);
            updateMutation.mutate(toUpdate);
        }
    };

    if (isLoading || !form) {
        return (
            <Layout>
                <div className="flex items-center justify-center min-h-[50vh]">
                    <p>Loading settings...</p>
                </div>
            </Layout>
        );
    }

    return (
        <Layout>
            <div className="space-y-6 max-w-4xl mx-auto">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight text-glow">Global Admin Settings</h1>
                    <p className="text-muted-foreground mt-1">Configure system-wide defaults and server settings.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className="glass-card">
                        <CardHeader>
                            <CardTitle>Email (SMTP) Settings</CardTitle>
                            <CardDescription>Used for invitations and recovery.</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="smtp_host">SMTP Host</Label>
                                <Input
                                    id="smtp_host"
                                    value={form.smtp_host}
                                    onChange={(e) => setForm({ ...form, smtp_host: e.target.value })}
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="smtp_port">SMTP Port</Label>
                                <Input
                                    id="smtp_port"
                                    type="number"
                                    value={form.smtp_port}
                                    onChange={(e) =>
                                        setForm({
                                            ...form,
                                            smtp_port: parseIntegerOrFallback(e.target.value, form.smtp_port),
                                        })
                                    }
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="smtp_user">SMTP User</Label>
                                <Input
                                    id="smtp_user"
                                    value={form.smtp_user}
                                    onChange={(e) => setForm({ ...form, smtp_user: e.target.value })}
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="smtp_password">
                                    SMTP Password
                                    {form.smtp_password_set && (
                                        <span className="text-xs text-muted-foreground ml-2">(currently set)</span>
                                    )}
                                </Label>
                                <Input
                                    id="smtp_password"
                                    type="password"
                                    value={smtpPassword}
                                    onChange={(e) => setSmtpPassword(e.target.value)}
                                    placeholder={form.smtp_password_set ? "Leave empty to keep current" : "Password"}
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="smtp_from">From Email</Label>
                                <Input
                                    id="smtp_from"
                                    value={form.smtp_from_email || ""}
                                    onChange={(e) => setForm({ ...form, smtp_from_email: e.target.value })}
                                    placeholder="noreply@tokemizer.com"
                                />
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="glass-card">
                        <CardHeader>
                            <CardTitle>Stripe Payment Configuration</CardTitle>
                            <CardDescription>Configure Stripe for subscription billing.</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="stripe_publishable">Stripe Publishable Key</Label>
                                <Input
                                    id="stripe_publishable"
                                    value={form.stripe_publishable_key}
                                    onChange={(e) => setForm({ ...form, stripe_publishable_key: e.target.value })}
                                    placeholder="pk_live_..."
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="stripe_secret">
                                    Stripe Secret Key
                                    {form.stripe_secret_key_set && (
                                        <span className="text-xs text-muted-foreground ml-2">(currently set)</span>
                                    )}
                                </Label>
                                <Input
                                    id="stripe_secret"
                                    type="password"
                                    value={stripeSecretKey}
                                    onChange={(e) => setStripeSecretKey(e.target.value)}
                                    placeholder={
                                        form.stripe_secret_key_set
                                            ? "Leave empty to keep current key"
                                            : "sk_live_..."
                                    }
                                />
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="glass-card">
                        <CardHeader>
                            <CardTitle>Security Configuration</CardTitle>
                            <CardDescription>Manage session lifetimes and CORS.</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="access_token_expiry">Access Token (Minutes)</Label>
                                    <Input
                                        id="access_token_expiry"
                                        type="number"
                                        value={form.access_token_expire_minutes || 30}
                                        onChange={(e) =>
                                            setForm({
                                                ...form,
                                                access_token_expire_minutes: parseIntegerOrFallback(
                                                    e.target.value,
                                                    form.access_token_expire_minutes || 30,
                                                ),
                                            })
                                        }
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="refresh_token_expiry">Refresh Token (Days)</Label>
                                    <Input
                                        id="refresh_token_expiry"
                                        type="number"
                                        value={form.refresh_token_expire_days || 7}
                                        onChange={(e) =>
                                            setForm({
                                                ...form,
                                                refresh_token_expire_days: parseIntegerOrFallback(
                                                    e.target.value,
                                                    form.refresh_token_expire_days || 7,
                                                ),
                                            })
                                        }
                                    />
                                </div>
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="cors_origins">CORS Origins</Label>
                                <Input
                                    id="cors_origins"
                                    value={form.cors_origins || "*"}
                                    onChange={(e) => setForm({ ...form, cors_origins: e.target.value })}
                                    placeholder="https://app.example.com, http://localhost:3000"
                                />
                                <p className="text-xs text-muted-foreground">
                                    Comma-separated list of allowed origins.
                                </p>
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="glass-card">
                        <CardHeader>
                            <CardTitle>System & Optimization</CardTitle>
                            <CardDescription>Configure core system behavior.</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="flex flex-col space-y-4">
                                <div className="flex items-center justify-between">
                                    <div className="space-y-0.5">
                                        <Label>Optimization History</Label>
                                        <p className="text-xs text-muted-foreground">
                                            Record all optimization requests to DB
                                        </p>
                                    </div>
                                    <Switch 
                                        checked={form.history_enabled !== false} // Default to true
                                        onCheckedChange={(checked) => setForm({ ...form, history_enabled: checked })}
                                    />
                                </div>
                                <div className="flex items-center justify-between">
                                    <div className="space-y-0.5">
                                        <Label>Pre-warm Models</Label>
                                        <p className="text-xs text-muted-foreground">
                                            Load ML models into memory on startup
                                        </p>
                                    </div>
                                    <Switch 
                                        checked={form.optimizer_prewarm_models !== false} // Default to true
                                        onCheckedChange={(checked) =>
                                            setForm({ ...form, optimizer_prewarm_models: checked })
                                        }
                                    />
                                </div>
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="glass-card">
                        <CardHeader>
                            <CardTitle>Logging & Telemetry</CardTitle>
                            <CardDescription>Configure application behavior and observability.</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="log_level">Log Level</Label>
                                <Select 
                                    value={form.log_level || "INFO"} 
                                    onValueChange={(val) => setForm({ ...form, log_level: val })}
                                >
                                    <SelectTrigger id="log_level">
                                        <SelectValue placeholder="Select Log Level" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="DEBUG">DEBUG</SelectItem>
                                        <SelectItem value="INFO">INFO</SelectItem>
                                        <SelectItem value="WARNING">WARNING</SelectItem>
                                        <SelectItem value="ERROR">ERROR</SelectItem>
                                    </SelectContent>
                                </Select>
                                <p className="text-xs text-muted-foreground">Verbosity of backend logs.</p>
                            </div>

                            <div className="flex flex-col space-y-2">
                                <Label>Telemetry</Label>
                                <div className="flex items-center space-x-2">
                                    <Switch 
                                        checked={form.telemetry_enabled || false}
                                        onCheckedChange={(checked) => setForm({ ...form, telemetry_enabled: checked })}
                                    />
                                    <span className="text-sm text-muted-foreground">
                                        Enable global performance telemetry collection
                                    </span>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </div>

                <div className="flex justify-end gap-3">
                    <Button
                        variant="outline"
                        onClick={() => {
                            if (data) {
                                setForm(data);
                                setStripeSecretKey("");
                                setSmtpPassword("");
                            }
                        }}
                        tooltip="Reset all settings to current saved values"
                    >
                        Reset
                    </Button>
                    <Button
                        onClick={handleSave}
                        disabled={updateMutation.isPending}
                        tooltip={
                            updateMutation.isPending
                                ? "Saving settings..."
                                : "Save admin settings changes"
                        }
                    >
                        {updateMutation.isPending ? "Saving..." : "Save Settings"}
                    </Button>
                </div>
            </div>
        </Layout>
    );
}
