import { useEffect, useState } from "react";
import { PlanCard } from "./PlanCard";
import { toApiUrl } from "@/lib/apiUrl";

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

interface AvailablePlansProps {
    currentPlan?: string;
    onSelectPlan: (planId: string) => void;
}

export function AvailablePlans({ currentPlan, onSelectPlan }: AvailablePlansProps) {
    const [plans, setPlans] = useState<Plan[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchPlans = async () => {
            try {
                const response = await fetch(toApiUrl("/api/auth/plans"));
                if (!response.ok) {
                    throw new Error("Failed to fetch plans");
                }
                const data = await response.json();
                setPlans(data);
            } catch (err) {
                setError(err instanceof Error ? err.message : "An error occurred");
            } finally {
                setLoading(false);
            }
        };

        fetchPlans();
    }, []);

    if (loading) {
        return (
            <div className="space-y-4">
                <h2 className="text-2xl font-bold">Available Plans</h2>
                <div className="flex gap-4 overflow-x-auto pb-4">
                    {[1, 2, 3, 4].map((i) => (
                        <div
                            key={i}
                            className="min-w-[320px] h-96 bg-muted/50 rounded-lg animate-pulse"
                        />
                    ))}
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="space-y-4">
                <h2 className="text-2xl font-bold">Available Plans</h2>
                <div className="rounded-lg border border-destructive/50 bg-white p-4 text-sm text-destructive">
                    Failed to load plans: {error}
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-4">
            <h2 className="text-2xl font-bold">Available Plans</h2>
            <div className="flex gap-4 overflow-x-auto pb-4 -mx-6 px-6">
                {plans.length === 0 ? (
                    <div className="text-muted-foreground">No plans available</div>
                ) : (
                    plans.map((plan) => (
                        <PlanCard
                            key={plan.id}
                            plan={plan}
                            currentPlan={currentPlan}
                            onSelect={onSelectPlan}
                        />
                    ))
                )}
            </div>
        </div>
    );
}
