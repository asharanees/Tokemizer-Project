import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, Zap } from "lucide-react";
import { HelpTooltip } from "@/components/ui/HelpTooltip";

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

interface PlanCardProps {
    plan: Plan;
    currentPlan?: string;
    onSelect: (planId: string) => void;
}

export function PlanCard({ plan, currentPlan, onSelect }: PlanCardProps) {
    const monthlyPrice = plan.monthly_price_cents / 100;
    const annualPrice = plan.annual_price_cents ? plan.annual_price_cents / 100 : null;
    const isContactSalesPlan = plan.monthly_price_cents < 0 || (plan.annual_price_cents ?? 0) < 0;
    const monthlyDiscount = plan.monthly_discount_percent || 0;
    const yearlyDiscount = plan.yearly_discount_percent || 0;
    
    const discountedMonthlyPrice = monthlyPrice * (1 - monthlyDiscount / 100);
    const discountedAnnualPrice = annualPrice ? annualPrice * (1 - yearlyDiscount / 100) : null;
    
    const isCurrentPlan = currentPlan === plan.id;
    
    const renderPrice = () => {
        if (isContactSalesPlan) {
            return (
                <div className="flex items-center gap-2">
                    <span className="text-lg font-semibold">Contact Sales</span>
                    <HelpTooltip content="A price of -1 means this plan is not free and requires contacting sales." />
                </div>
            );
        }

        if (monthlyPrice === 0) {
            return <span className="text-lg font-semibold">Free</span>;
        }
        
        return (
            <div className="space-y-1">
                <div className="flex items-center gap-2">
                    {monthlyDiscount > 0 ? (
                        <>
                            <span className="text-sm text-muted-foreground line-through">${monthlyPrice.toFixed(2)}</span>
                            <span className="text-lg font-semibold text-green-600">${discountedMonthlyPrice.toFixed(2)}</span>
                            <Badge variant="secondary" className="text-xs">{monthlyDiscount}% off</Badge>
                        </>
                    ) : (
                        <span className="text-lg font-semibold">${monthlyPrice.toFixed(2)}</span>
                    )}
                </div>
                <div className="text-xs text-muted-foreground">/month</div>
                
                {annualPrice && (
                    <div className="pt-2 border-t border-border">
                        <div className="flex items-center gap-2">
                            {yearlyDiscount > 0 ? (
                                <>
                                    <span className="text-xs text-muted-foreground line-through">${annualPrice.toFixed(2)}</span>
                                    <span className="text-sm font-semibold text-green-600">${discountedAnnualPrice!.toFixed(2)}</span>
                                    <Badge variant="secondary" className="text-xs">{yearlyDiscount}% off</Badge>
                                </>
                            ) : (
                                <span className="text-sm font-semibold">${annualPrice.toFixed(2)}</span>
                            )}
                        </div>
                        <div className="text-xs text-muted-foreground">/year</div>
                    </div>
                )}
            </div>
        );
    };

    return (
        <Card className={`relative flex flex-col ${isCurrentPlan ? "border-primary shadow-md" : ""}`}>
            {isCurrentPlan && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-primary text-primary-foreground text-xs font-medium rounded-full">
                    Current Plan
                </div>
            )}
            <CardHeader>
                <CardTitle className="flex justify-between items-start">
                    <span>{plan.name}</span>
                </CardTitle>
                <CardDescription>
                    {plan.description || `${plan.monthly_quota.toLocaleString()} requests / month`}
                </CardDescription>
            </CardHeader>
            <CardContent className="flex-1 space-y-4">
                <div className="mb-4">
                    {renderPrice()}
                </div>
                <div className="space-y-2">
                    <div className="space-y-1">
                        <div className="text-sm text-muted-foreground">API Quota</div>
                        <div className="text-2xl font-semibold">
                            {plan.monthly_quota === -1 ? "Unlimited" : plan.monthly_quota.toLocaleString()}
                        </div>
                        {plan.monthly_quota !== -1 && (
                            <div className="text-xs text-muted-foreground">calls per month</div>
                        )}
                    </div>

                    <div className="space-y-1">
                        <div className="text-sm text-muted-foreground">Rate Limit</div>
                        <div className="text-lg font-semibold flex items-center gap-2">
                            <Zap className="w-4 h-4 text-yellow-500" />
                            {plan.rate_limit_rpm} req/min
                        </div>
                    </div>

                    <div className="space-y-1">
                        <div className="text-sm text-muted-foreground">API Keys</div>
                        <div className="text-lg font-semibold">
                            {plan.max_api_keys === -1 ? "Unlimited" : plan.max_api_keys}
                        </div>
                    </div>
                </div>

                {plan.features.length > 0 && (
                    <div className="rounded-lg bg-muted/50 p-3 space-y-2">
                        {plan.features.map((feature) => (
                            <div key={feature} className="flex items-start gap-2 text-sm">
                                <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
                                <span>{feature}</span>
                            </div>
                        ))}
                    </div>
                )}
            </CardContent>

            <CardFooter>
                <Button
                    onClick={() => onSelect(plan.id)}
                    disabled={isCurrentPlan}
                    className="w-full"
                    variant={isCurrentPlan ? "secondary" : "default"}
                    tooltip={
                        isCurrentPlan
                            ? "This is your current plan"
                            : isContactSalesPlan
                                ? "A -1 price indicates enterprise contact-sales pricing"
                                : monthlyPrice === 0
                                    ? "Downgrade to free plan"
                                    : "Upgrade to this plan"
                    }
                >
                    {isCurrentPlan
                        ? "Current Plan"
                        : isContactSalesPlan
                            ? "Contact Sales"
                            : monthlyPrice === 0
                                ? "Downgrade"
                                : "Upgrade"}
                </Button>
            </CardFooter>
        </Card>
    );
}
