import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogDescription } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { authFetch } from "@/lib/authFetch";

interface Plan {
    id: string;
    name: string;
    description?: string;
    monthly_price_cents: number;
    annual_price_cents?: number;
    monthly_quota: number;
    rate_limit_rpm: number;
    concurrent_optimization_jobs: number;
    batch_size_limit: number;
    optimization_history_retention_days: number;
    telemetry_retention_days: number;
    audit_log_retention_days: number;
    custom_canonical_mappings_limit: number;
    max_api_keys: number;
    features: string[];
    is_active: boolean;
    is_public: boolean;
    plan_term: string;
    monthly_discount_percent: number;
    yearly_discount_percent: number;
}

interface PlanForm extends Plan {
    monthly_price_display: string;
    annual_price_display: string;
}

export default function Plans() {
    const { toast } = useToast();
    const queryClient = useQueryClient();
    const [selectedPlan, setSelectedPlan] = useState<PlanForm | null>(null);
    const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
    const [planToDelete, setPlanToDelete] = useState<Plan | null>(null);
    const [showPublicOnly, setShowPublicOnly] = useState(false);

    const { data: plans, isLoading } = useQuery<Plan[]>({
        queryKey: ["admin-plans", { publicOnly: showPublicOnly }],
        queryFn: async () => {
            const query = showPublicOnly ? "?public_only=true" : "";
            const res = await authFetch(`/api/admin/plans${query}`);
            if (!res.ok) throw new Error("Failed to fetch plans");
            return res.json();
        },
    });

    const saveMutation = useMutation({
        mutationFn: async (plan: PlanForm) => {
            const payload: Partial<PlanForm> = { ...plan };
            delete payload.monthly_price_display;
            delete payload.annual_price_display;
            if (!payload.id) {
                delete payload.id;
            }
            const res = await authFetch("/api/admin/plans", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!res.ok) throw new Error("Failed to save plan");
            return res.json();
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["admin-plans"] });
            setIsEditDialogOpen(false);
            toast({ title: "Plan saved successfully" });
        },
        onError: (error: Error) => {
            toast({ title: "Error", description: error.message, variant: "destructive" });
        },
    });

    const deleteMutation = useMutation({
        mutationFn: async (planId: string) => {
            const res = await authFetch(`/api/admin/plans/${planId}`, {
                method: "DELETE",
            });
            if (!res.ok) {
                const error = await res.json();
                throw new Error(error.detail || "Failed to delete plan");
            }
            return res.json();
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["admin-plans"] });
            setIsDeleteDialogOpen(false);
            setPlanToDelete(null);
            toast({ title: "Plan deleted successfully" });
        },
        onError: (error: Error) => {
            toast({ title: "Error", description: error.message, variant: "destructive" });
        },
    });

    const handleEdit = (plan: Plan) => {
        setSelectedPlan({
            ...plan,
            monthly_price_display: (plan.monthly_price_cents / 100).toFixed(2),
            annual_price_display: plan.annual_price_cents ? (plan.annual_price_cents / 100).toFixed(2) : "",
        });
        setIsEditDialogOpen(true);
    };

    const handleDelete = (plan: Plan) => {
        setPlanToDelete(plan);
        setIsDeleteDialogOpen(true);
    };

    const confirmDelete = () => {
        if (planToDelete) {
            deleteMutation.mutate(planToDelete.id);
        }
    };

    const handleCreate = () => {
        setSelectedPlan({
            id: "",
            name: "",
            description: "",
            monthly_price_cents: 0,
            annual_price_cents: 0,
            monthly_quota: 1000,
            rate_limit_rpm: 1000,
            concurrent_optimization_jobs: 5,
            batch_size_limit: 1000,
            optimization_history_retention_days: 365,
            telemetry_retention_days: 365,
            audit_log_retention_days: 365,
            custom_canonical_mappings_limit: 1000,
            max_api_keys: 5,
            features: [],
            is_active: true,
            is_public: true,
            plan_term: "monthly",
            monthly_discount_percent: 0,
            yearly_discount_percent: 0,
            monthly_price_display: "0.00",
            annual_price_display: "",
        });
        setIsEditDialogOpen(true);
    };

    const handlePriceChange = (type: 'monthly' | 'annual', value: string) => {
        if (!selectedPlan) return;
        
        const updates: Partial<PlanForm> = {};
        if (type === 'monthly') {
            updates.monthly_price_display = value;
            const floatVal = parseFloat(value);
            if (!isNaN(floatVal)) {
                updates.monthly_price_cents = Math.round(floatVal * 100);
            }
        } else {
            updates.annual_price_display = value;
            const floatVal = parseFloat(value);
            if (!isNaN(floatVal)) {
                updates.annual_price_cents = Math.round(floatVal * 100);
            } else {
                updates.annual_price_cents = 0;
            }
        }
        
        setSelectedPlan({ ...selectedPlan, ...updates });
    };

    const formatPriceDisplay = (cents?: number) => {
        if (cents === undefined || cents === null) return "-";
        if (cents < 0) return "Contact Sales";
        return `$${(cents / 100).toFixed(2)}`;
    };


    return (
        <Layout>
            <div className="space-y-6">
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight">Subscription Plans</h1>
                        <p className="text-muted-foreground">Define and manage plans available to customers.</p>
                    </div>
                    <div className="flex items-center gap-3">
                        <label className="flex items-center gap-2 text-sm text-muted-foreground">
                            <input
                                type="checkbox"
                                checked={showPublicOnly}
                                onChange={(e) => setShowPublicOnly(e.target.checked)}
                            />
                            Public plans only
                        </label>
                        <Button onClick={handleCreate} tooltip="Create a new subscription plan">Add New Plan</Button>
                    </div>
                </div>

                <Card className="glass-card">
                    <CardHeader>
                        <CardTitle>All Plans</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>ID</TableHead>
                                    <TableHead>Name</TableHead>
                                    <TableHead>Monthly Price</TableHead>
                                    <TableHead>Annual Price</TableHead>
                                    <TableHead>Plan Term</TableHead>
                                    <TableHead>Monthly Quota</TableHead>
                                    <TableHead>API Keys</TableHead>
                                    <TableHead>Batch/Concurrency</TableHead>
                                    <TableHead>Visibility</TableHead>
                                    <TableHead>Status</TableHead>
                                    <TableHead className="text-right">Actions</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {isLoading ? (
                                    <TableRow>
                                        <TableCell colSpan={11} className="text-center">Loading...</TableCell>
                                    </TableRow>
                                ) : (
                                    plans?.map((plan) => (
                                        <TableRow key={plan.id}>
                                            <TableCell className="font-mono text-xs">{plan.id}</TableCell>
                                            <TableCell className="font-medium">
                                                <div className="flex flex-col">
                                                    <span>{plan.name}</span>
                                                    {plan.description && (
                                                        <span className="text-xs text-muted-foreground">{plan.description}</span>
                                                    )}
                                                </div>
                                            </TableCell>
                                            <TableCell>{formatPriceDisplay(plan.monthly_price_cents)}</TableCell>
                                            <TableCell>{formatPriceDisplay(plan.annual_price_cents)}</TableCell>
                                            <TableCell className="capitalize">{plan.plan_term}</TableCell>
                                            <TableCell>{plan.monthly_quota.toLocaleString()}</TableCell>
                                            <TableCell>{plan.max_api_keys}</TableCell>
                                            <TableCell>{plan.batch_size_limit}/{plan.concurrent_optimization_jobs}</TableCell>
                                            <TableCell>
                                                <Badge variant={plan.is_public ? "outline" : "secondary"}>
                                                    {plan.is_public ? "Public" : "Private"}
                                                </Badge>
                                            </TableCell>
                                            <TableCell>
                                                <Badge variant={plan.is_active ? "outline" : "secondary"}>
                                                    {plan.is_active ? "Active" : "Inactive"}
                                                </Badge>
                                            </TableCell>
                                            <TableCell className="text-right space-x-2">
                                                <Button variant="ghost" size="sm" onClick={() => handleEdit(plan)} tooltip="Edit subscription plan">
                                                    Edit
                                                </Button>
                                                <Button
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={() => handleDelete(plan)}
                                                    className="text-red-600 hover:text-red-700 hover:bg-red-50"
                                                    tooltip="Delete subscription plan"
                                                >
                                                    Delete
                                                </Button>
                                            </TableCell>
                                        </TableRow>
                                    ))
                                )}
                            </TableBody>
                        </Table>
                    </CardContent>
                </Card>
            </div>

            <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
                <DialogContent className="max-w-2xl">
                    <DialogHeader>
                        <DialogTitle>{selectedPlan?.id ? "Edit Plan" : "Create New Plan"}</DialogTitle>
                        <DialogDescription>
                            Configure plan details and limits. ID is system-generated.
                        </DialogDescription>
                    </DialogHeader>
                    {selectedPlan && (
                        <div className="grid grid-cols-2 gap-4 py-4">
                            {selectedPlan.id && (
                                <div className="space-y-2">
                                    <Label>Plan ID</Label>
                                    <div className="text-sm font-mono text-muted-foreground bg-muted p-2 rounded">{selectedPlan.id}</div>
                                </div>
                            )}
                            <div className="space-y-2">
                                <Label htmlFor="plan-name">Display Name</Label>
                                <Input
                                    id="plan-name"
                                    value={selectedPlan.name}
                                    onChange={(e) => setSelectedPlan({ ...selectedPlan, name: e.target.value })}
                                />
                            </div>
                            <div className="col-span-2 space-y-2">
                                <Label htmlFor="plan-description">Description</Label>
                                <Textarea
                                    id="plan-description"
                                    value={selectedPlan.description || ""}
                                    onChange={(e) => setSelectedPlan({ ...selectedPlan, description: e.target.value })}
                                    placeholder="Optional description of the plan"
                                    rows={2}
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="plan-term">Plan Term</Label>
                                <select
                                    id="plan-term"
                                    value={selectedPlan.plan_term}
                                    onChange={(e) => setSelectedPlan({ ...selectedPlan, plan_term: e.target.value })}
                                    className="w-full px-3 py-2 border border-input rounded-md bg-background text-sm"
                                >
                                    <option value="monthly">Monthly</option>
                                    <option value="yearly">Yearly</option>
                                    <option value="both">Both (Monthly & Yearly)</option>
                                </select>
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="plan-quota">Monthly Token Quota</Label>
                                <Input
                                    id="plan-quota"
                                    type="number"
                                    value={selectedPlan.monthly_quota}
                                    onChange={(e) => setSelectedPlan({ ...selectedPlan, monthly_quota: parseInt(e.target.value) })}
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="monthly-price">Monthly Price ($)</Label>
                                <Input
                                    id="monthly-price"
                                    type="text"
                                    inputMode="decimal"
                                    value={selectedPlan.monthly_price_display}
                                    onChange={(e) => handlePriceChange('monthly', e.target.value)}
                                    placeholder="0.00"
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="annual-price">Annual Price ($)</Label>
                                <Input
                                    id="annual-price"
                                    type="text"
                                    inputMode="decimal"
                                    value={selectedPlan.annual_price_display}
                                    onChange={(e) => handlePriceChange('annual', e.target.value)}
                                    placeholder="Leave empty for monthly only"
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="monthly-discount">Monthly Term Discount (%)</Label>
                                <Input
                                    id="monthly-discount"
                                    type="number"
                                    min="0"
                                    max="100"
                                    value={selectedPlan.monthly_discount_percent}
                                    onChange={(e) => setSelectedPlan({ ...selectedPlan, monthly_discount_percent: parseInt(e.target.value) || 0 })}
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="yearly-discount">Yearly Term Discount (%)</Label>
                                <Input
                                    id="yearly-discount"
                                    type="number"
                                    min="0"
                                    max="100"
                                    value={selectedPlan.yearly_discount_percent}
                                    onChange={(e) => setSelectedPlan({ ...selectedPlan, yearly_discount_percent: parseInt(e.target.value) || 0 })}
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="plan-rpm">Rate Limit (RPM)</Label>
                                <Input
                                    id="plan-rpm"
                                    type="number"
                                    value={selectedPlan.rate_limit_rpm}
                                    onChange={(e) => setSelectedPlan({ ...selectedPlan, rate_limit_rpm: parseInt(e.target.value) })}
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="plan-keys">Max API Keys</Label>
                                <Input
                                    id="plan-keys"
                                    type="number"
                                    value={selectedPlan.max_api_keys}
                                    onChange={(e) => setSelectedPlan({ ...selectedPlan, max_api_keys: parseInt(e.target.value) })}
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="plan-concurrency">Concurrent Optimization Jobs</Label>
                                <Input id="plan-concurrency" type="number" value={selectedPlan.concurrent_optimization_jobs} onChange={(e) => setSelectedPlan({ ...selectedPlan, concurrent_optimization_jobs: parseInt(e.target.value) })} />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="plan-batch">Batch Size Limit</Label>
                                <Input id="plan-batch" type="number" value={selectedPlan.batch_size_limit} onChange={(e) => setSelectedPlan({ ...selectedPlan, batch_size_limit: parseInt(e.target.value) })} />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="plan-canonical-limit">Custom Canonical Mappings Limit</Label>
                                <Input id="plan-canonical-limit" type="number" value={selectedPlan.custom_canonical_mappings_limit} onChange={(e) => setSelectedPlan({ ...selectedPlan, custom_canonical_mappings_limit: parseInt(e.target.value) })} />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="plan-history-retention">History Retention (days)</Label>
                                <Input id="plan-history-retention" type="number" value={selectedPlan.optimization_history_retention_days} onChange={(e) => setSelectedPlan({ ...selectedPlan, optimization_history_retention_days: parseInt(e.target.value) })} />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="plan-telemetry-retention">Telemetry Retention (days)</Label>
                                <Input id="plan-telemetry-retention" type="number" value={selectedPlan.telemetry_retention_days} onChange={(e) => setSelectedPlan({ ...selectedPlan, telemetry_retention_days: parseInt(e.target.value) })} />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="plan-audit-retention">Audit Retention (days)</Label>
                                <Input id="plan-audit-retention" type="number" value={selectedPlan.audit_log_retention_days} onChange={(e) => setSelectedPlan({ ...selectedPlan, audit_log_retention_days: parseInt(e.target.value) })} />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="plan-public">Public Plan</Label>
                                <div className="flex items-center gap-2 h-10">
                                    <input
                                        id="plan-public"
                                        type="checkbox"
                                        checked={selectedPlan.is_public}
                                        onChange={(e) => setSelectedPlan({ ...selectedPlan, is_public: e.target.checked })}
                                    />
                                    <span className="text-sm text-muted-foreground">Enable self-service visibility</span>
                                </div>
                            </div>
                            <div className="col-span-2 space-y-2">
                                <Label htmlFor="plan-features">Features (JSON Array of strings)</Label>
                                <Textarea
                                    id="plan-features"
                                    rows={4}
                                    value={JSON.stringify(selectedPlan.features, null, 2)}
                                    onChange={(e) => {
                                        try {
                                            setSelectedPlan({ ...selectedPlan, features: JSON.parse(e.target.value) });
                                        } catch {
                                            // Allow editing invalid JSON
                                        }
                                    }}
                                />
                            </div>
                        </div>
                    )}
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setIsEditDialogOpen(false)} tooltip="Cancel editing and discard changes">Cancel</Button>
                        <Button
                            onClick={() => selectedPlan && saveMutation.mutate(selectedPlan)}
                            disabled={saveMutation.isPending}
                            tooltip={saveMutation.isPending ? "Saving plan..." : "Save subscription plan changes"}
                        >
                            {saveMutation.isPending ? "Saving..." : "Save Plan"}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Delete Plan</DialogTitle>
                        <DialogDescription>
                            Are you sure you want to delete the plan "{planToDelete?.name}"? This action cannot be undone. The plan can only be deleted if no customers have active subscriptions to it.
                        </DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button
                            variant="outline"
                            onClick={() => setIsDeleteDialogOpen(false)}
                            disabled={deleteMutation.isPending}
                            tooltip="Cancel deletion and close dialog"
                        >
                            Cancel
                        </Button>
                        <Button
                            variant="destructive"
                            onClick={confirmDelete}
                            disabled={deleteMutation.isPending}
                            tooltip={deleteMutation.isPending ? "Deleting plan..." : "Permanently delete this subscription plan"}
                        >
                            {deleteMutation.isPending ? "Deleting..." : "Delete Plan"}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </Layout>
    );
}
