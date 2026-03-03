import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { CheckCircle2, Zap, Settings } from "lucide-react";
import { AvailablePlans } from "@/components/dashboard/AvailablePlans";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { authFetch } from "@/lib/authFetch";


interface UsageData {
    calls_used: number;
    quota_limit: number;
    remaining: number;
    quota_overage_bonus?: number;
    plan_limits?: {
        rate_limit_rpm: number;
        concurrent_optimization_jobs: number;
        batch_size_limit: number;
        optimization_history_retention_days: number;
        telemetry_retention_days: number;
        audit_log_retention_days: number;
        custom_canonical_mappings_limit: number;
    };
}

interface Plan {
    id: string;
    name: string;
    description?: string;
    monthly_price_cents: number;
    annual_price_cents?: number;
    monthly_quota: number;
    rate_limit_rpm: number;
    max_api_keys: number;
    features: string[];
    is_active: boolean;
    plan_term?: string;
    monthly_discount_percent?: number;
    yearly_discount_percent?: number;
}

export default function Subscription() {
    const { user } = useAuth();
    const [selectedPlanId, setSelectedPlanId] = useState<string | null>(null);
    const [selectedPlan, setSelectedPlan] = useState<Plan | null>(null);
    const [isUpgradeDialogOpen, setIsUpgradeDialogOpen] = useState(false);
    const [isManagingSubscription, setIsManagingSubscription] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);

    const usageQuery = useQuery<UsageData>({
        queryKey: ["usage"],
        queryFn: async () => {
            const res = await authFetch("/api/usage");
            if (!res.ok) {
                throw new Error("Failed to load usage");
            }
            return res.json();
        },
        refetchInterval: 30000,
        enabled: !!user,
    });

    if (!user) return null;

    const usageData = usageQuery.data;
    const quotaLimit = usageData?.quota_limit ?? 0;
    const usagePercentage =
        quotaLimit > 0 && usageData
            ? Math.min(100, Math.max(0, (usageData.calls_used / quotaLimit) * 100))
            : 0;
    const callsLabel = usageData
        ? `${usageData.calls_used.toLocaleString()} / ${usageData.quota_limit.toLocaleString()}`
        : usageQuery.isError
            ? "Usage unavailable"
            : "Loading usage...";
    const remainingCount = usageData ? Math.max(0, usageData.remaining) : 0;
    const remainingLabel = usageData
        ? `${remainingCount.toLocaleString()} calls remaining`
        : usageQuery.isError
            ? "Usage unavailable"
            : "Loading usage...";

    const handleManageSubscription = async () => {
        setIsManagingSubscription(true);
        try {
            // TODO: Implement redirect to Stripe Customer Portal
            // For now, show a placeholder message
            const response = await authFetch("/api/billing/portal-session", {
                method: "POST",
            });

            if (response.ok) {
                const { url } = await response.json();
                window.location.href = url;
            } else {
                console.error("Failed to create portal session");
                alert("Unable to access subscription management. Please try again later.");
            }
        } catch (error) {
            console.error("Portal error:", error);
            alert("Error accessing subscription management portal.");
        } finally {
            setIsManagingSubscription(false);
        }
    };

    const handleSelectPlan = async (planId: string) => {
        if (planId === user.subscription_tier) {
            return; // Already on this plan
        }

        // Fetch plan details to check if payment is required
        try {
            const response = await authFetch("/api/auth/plans");
            if (!response.ok) throw new Error("Failed to fetch plan details");
            
            const plans: Plan[] = await response.json();
            const plan = plans.find((p) => p.id === planId);
            
            if (!plan) throw new Error("Plan not found");

            setSelectedPlan(plan);
            setSelectedPlanId(planId);
            setIsUpgradeDialogOpen(true);
        } catch (error) {
            console.error("Error fetching plan:", error);
        }
    };

    const handleConfirmUpgrade = async () => {
        if (!selectedPlanId || !selectedPlan) return;

        setIsProcessing(true);

        try {
            const requiresContactSales =
                selectedPlan.monthly_price_cents < 0 || (selectedPlan.annual_price_cents ?? 0) < 0;
            const hasCharge = selectedPlan.monthly_price_cents > 0;

            if (requiresContactSales) {
                window.location.href = "/contact";
                return;
            }

            if (hasCharge) {
                // Redirect to Stripe checkout for paid plan
                const response = await authFetch("/api/billing/checkout-session", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        plan_id: selectedPlanId,
                        success_url: `${window.location.origin}/subscription?checkout=success`,
                        cancel_url: `${window.location.origin}/subscription?checkout=cancel`,
                    }),
                });

                if (!response.ok) throw new Error("Failed to create checkout session");

                const { url } = await response.json();
                window.location.href = url;
            } else {
                // Free or trial plan - update directly without payment
                const response = await authFetch("/api/subscription/upgrade", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ plan_id: selectedPlanId }),
                });

                if (!response.ok) throw new Error("Failed to upgrade plan");

                setIsUpgradeDialogOpen(false);
                window.location.reload(); // Refresh to show updated plan
            }
        } catch (error) {
            console.error("Upgrade failed:", error);
            alert("Failed to process upgrade. Please try again.");
        } finally {
            setIsProcessing(false);
        }
    };

    return (
        <Layout>
            <div className="space-y-6 max-w-6xl">
                <h1 className="text-3xl font-bold tracking-tight font-display text-glow">Subscription</h1>

                <div className="grid gap-6 md:grid-cols-2">
                    <Card className="glass-card">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <Settings className="w-5 h-5" />
                                Current Plan
                            </CardTitle>
                            <CardDescription>You are currently on the <span className="font-semibold text-primary capitalize">{user.subscription_tier}</span> plan.</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="flex items-center justify-between">
                                <span className="text-sm font-medium text-muted-foreground">Subscription Status</span>
                                <Badge variant={user.subscription_status === "active" ? "default" : "secondary"}>
                                    {user.subscription_status === "active" ? "Active" : "Inactive"}
                                </Badge>
                            </div>

                            <div className="pt-2 border-t border-border">
                                <p className="text-xs text-muted-foreground mb-3">
                                    Manage your billing, payment methods, and subscription preferences.
                                </p>
                            </div>
                        </CardContent>
                        <CardFooter>
                            <Button 
                                onClick={handleManageSubscription}
                                disabled={isManagingSubscription}
                                className="w-full"
                                variant="outline"
                                tooltip={isManagingSubscription ? "Loading subscription portal..." : "Open subscription management portal"}
                            >
                                {isManagingSubscription ? "Loading..." : "Manage Subscription"}
                            </Button>
                        </CardFooter>
                    </Card>

                    <Card className="glass-card">
                        <CardHeader>
                            <CardTitle>Usage</CardTitle>
                            <CardDescription>Your API usage for the current billing period.</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-6">
                            <div className="space-y-2">
                                <div className="flex items-center justify-between text-sm">
                                    <span>API Calls</span>
                                    <span className="text-muted-foreground">{callsLabel}</span>
                                </div>
                                <Progress value={usagePercentage} className="h-2" />
                            </div>

                            <div className="rounded-lg bg-muted/50 p-4 space-y-3">
                                <div className="text-sm">
                                    Free overage bonus: <span className="font-medium">{(usageData?.quota_overage_bonus ?? 0).toLocaleString()} calls/mo</span>
                                </div>
                                <div className="flex items-center gap-2 text-sm">
                                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                                    <span>Unlimited Optimization History</span>
                                </div>
                                <div className="flex items-center gap-2 text-sm">
                                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                                    <span>Priority Processing</span>
                                </div>
                                <div className="flex items-center gap-2 text-sm">
                                    <Zap className="w-4 h-4 text-yellow-500" />
                                    <span>{remainingLabel}</span>
                                </div>
                            </div>

                            {usageData?.plan_limits && (
                                <div className="rounded-lg border border-border/60 p-4">
                                    <div className="text-sm font-medium mb-2">Plan limits</div>
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-muted-foreground">
                                        <div>RPM: {usageData.plan_limits.rate_limit_rpm.toLocaleString()}</div>
                                        <div>Concurrent jobs: {usageData.plan_limits.concurrent_optimization_jobs}</div>
                                        <div>Batch size: {usageData.plan_limits.batch_size_limit.toLocaleString()}</div>
                                        <div>Custom mappings: {usageData.plan_limits.custom_canonical_mappings_limit.toLocaleString()}</div>
                                        <div>History retention: {usageData.plan_limits.optimization_history_retention_days} days</div>
                                        <div>Telemetry retention: {usageData.plan_limits.telemetry_retention_days} days</div>
                                    </div>
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </div>

                {/* Available Plans Section */}
                <div className="rounded-lg border border-primary/20 bg-gradient-to-r from-primary/5 to-transparent p-6">
                    <AvailablePlans currentPlan={user.subscription_tier} onSelectPlan={handleSelectPlan} />
                </div>

                {/* Upgrade Confirmation Dialog */}
                <Dialog open={isUpgradeDialogOpen} onOpenChange={setIsUpgradeDialogOpen}>
                    <DialogContent>
                        <DialogHeader>
                            <DialogTitle>Upgrade Plan</DialogTitle>
                            <DialogDescription>
                                You're about to upgrade to a new plan. This change will be effective immediately.
                            </DialogDescription>
                        </DialogHeader>
                        <div className="py-4">
                            {selectedPlan && (
                                <div className="space-y-3">
                                    <div>
                                        <p className="text-sm text-muted-foreground mb-1">Plan:</p>
                                        <p className="font-semibold capitalize">{selectedPlan.name}</p>
                                    </div>
                                    <div>
                                        <p className="text-sm text-muted-foreground mb-1">Monthly Quota:</p>
                                        <p className="font-semibold">{selectedPlan.monthly_quota > 0 ? selectedPlan.monthly_quota.toLocaleString() : "Unlimited"} calls</p>
                                    </div>
                                    {(selectedPlan.monthly_price_cents < 0 || (selectedPlan.annual_price_cents ?? 0) < 0) && (
                                        <div className="pt-3 border-t border-border">
                                            <p className="text-sm text-muted-foreground mb-1">Billing:</p>
                                            <p className="text-sm font-semibold">Contact Sales</p>
                                            <p className="text-xs text-muted-foreground mt-2">
                                                A price of -1 means this plan is not free and requires contacting sales.
                                            </p>
                                        </div>
                                    )}
                                    {selectedPlan.monthly_price_cents > 0 && (
                                        <div className="pt-3 border-t border-border">
                                            <p className="text-sm text-muted-foreground mb-1">Billing:</p>
                                            <p className="text-sm">
                                                ${(selectedPlan.monthly_price_cents / 100).toFixed(2)}/month
                                            </p>
                                            <p className="text-xs text-muted-foreground mt-2">
                                                You will be redirected to Stripe to complete payment.
                                            </p>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                        <DialogFooter>
                            <Button variant="outline" onClick={() => setIsUpgradeDialogOpen(false)} tooltip="Cancel upgrade and close dialog">
                                Cancel
                            </Button>
                            <Button
                                onClick={handleConfirmUpgrade}
                                disabled={isProcessing}
                                tooltip={
                                    isProcessing
                                        ? "Processing upgrade..."
                                        : selectedPlan
                                            ? (selectedPlan.monthly_price_cents < 0 || (selectedPlan.annual_price_cents ?? 0) < 0)
                                                ? "A -1 price indicates enterprise contact-sales pricing"
                                                : selectedPlan.monthly_price_cents > 0
                                                    ? "Proceed to payment gateway"
                                                    : "Upgrade to selected plan"
                                            : "Upgrade to selected plan"
                                }
                            >
                                {isProcessing
                                    ? "Processing..."
                                    : selectedPlan
                                        ? (selectedPlan.monthly_price_cents < 0 || (selectedPlan.annual_price_cents ?? 0) < 0)
                                            ? "Contact Sales"
                                            : selectedPlan.monthly_price_cents > 0
                                                ? "Proceed to Payment"
                                                : "Upgrade Plan"
                                        : "Upgrade Plan"}
                            </Button>
                        </DialogFooter>
                    </DialogContent>
                </Dialog>
            </div>
        </Layout>
    );
}
