import { useState, useEffect } from "react";
import { PublicLayout } from "@/components/layout/PublicLayout";
import { useSeo } from "@/lib/seo";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Check, Loader2, Zap, Star, Crown, Building2, ArrowRight } from "lucide-react";
import { Link } from "wouter";
import { Badge } from "@/components/ui/badge";
import { HelpTooltip } from "@/components/ui/HelpTooltip";
import { toApiUrl } from "@/lib/apiUrl";

interface Plan {
    id: string;
    name: string;
    description?: string;
    monthly_price_cents: number;
    annual_price_cents?: number;
    monthly_quota: number;
    max_api_keys: number;
    features: string[];
    is_active: boolean;
    is_public: boolean;
}

export default function Plan() {
    useSeo({
        title: "Pricing Plans",
        description: "Compare Tokemizer pricing plans. Start for free or choose enterprise options for high-volume prompt compression and cost reduction.",
        path: "/plans",
    });

    const [plans, setPlans] = useState<Plan[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchPlans = async () => {
            try {
                const response = await fetch(toApiUrl("/api/subscription/plans"));
                if (!response.ok) {
                    throw new Error("Failed to fetch plans");
                }
                const data = await response.json();
                setPlans(data.filter((p: Plan) => p.is_active && p.is_public));
            } catch (error) {
                console.error("Failed to fetch plans", error);
                setPlans([]);
            } finally {
                setLoading(false);
            }
        };
        fetchPlans();
    }, []);

    const formatPrice = (cents: number) => {
        if (cents === 0) return "Free";
        if (cents < 0) return "Custom";
        return `$${(cents / 100).toFixed(0)}`;
    };

    const isContactSalesPlan = (plan: Plan) =>
        plan.monthly_price_cents < 0 || (plan.annual_price_cents ?? 0) < 0;

    const getPlanIcon = (planName: string) => {
        const name = planName.toLowerCase();
        if (name.includes("enterprise")) return <Building2 className="h-5 w-5" />;
        if (name.includes("pro")) return <Crown className="h-5 w-5" />;
        return <Zap className="h-5 w-5" />;
    };

    const getPlanColor = () => "from-slate-600 to-slate-800";

    const getGettingStartedHref = (plan: Plan) => {
        if (isContactSalesPlan(plan)) return "/contact";
        const params = new URLSearchParams({
            action: "register",
            plan_id: plan.id,
        });
        return `/login?${params.toString()}`;
    };

    return (
        <PublicLayout>
            <div className="bg-background">
                {/* Hero Section */}
                <div className="bg-gradient-to-br from-primary/5 via-background to-secondary/5 py-16 lg:py-24">
                    <div className="container mx-auto px-4 text-center">
                        <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-2 rounded-full text-sm font-medium mb-6">
                            <Star className="h-4 w-4" />
                            Transparent Pricing
                        </div>
                        <h1 className="text-4xl font-extrabold tracking-tight lg:text-6xl mb-6 text-foreground">
                            Choose Your
                            <span className="text-primary"> Optimization</span> Plan
                        </h1>
                        <p className="text-xl text-muted-foreground max-w-3xl mx-auto mb-8">
                            Start free and scale as you grow. Reduce LLM cost upto 70% with our core optimization technology, enterprise-grade security, and support.
                        </p>
                        <div className="flex justify-center gap-4 text-sm text-muted-foreground">
                            <span className="flex items-center gap-2">
                                <Check className="h-4 w-4 text-green-500" />
                                No setup fees
                            </span>
                            <span className="flex items-center gap-2">
                                <Check className="h-4 w-4 text-green-500" />
                                Cancel anytime
                            </span>
                            <span className="flex items-center gap-2">
                                <Check className="h-4 w-4 text-green-500" />
                                14-day free trial
                            </span>
                        </div>
                    </div>
                </div>

                <div className="container mx-auto px-4 py-16 lg:py-24">
                    {loading ? (
                        <div className="flex justify-center py-20">
                            <Loader2 className="h-10 w-10 animate-spin text-primary" />
                        </div>
                    ) : plans.length === 0 ? (
                        <div className="text-center py-12">
                            <p className="text-muted-foreground">No active public plans are available right now. Please check back shortly.</p>
                        </div>
                    ) : (
                        <div className="grid md:grid-cols-3 gap-8 max-w-7xl mx-auto">
                            {plans.map((plan) => {
                                const isPro = plan.name.toLowerCase().includes("pro");
                                const isEnterprise = plan.name.toLowerCase().includes("enterprise");
                                return (
                                    <Card key={plan.id} className="relative flex h-full flex-col border-border/60 bg-card/60 shadow-md transition-all duration-300 hover:shadow-xl">
                                        {isPro && (
                                            <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                                                <Badge className="bg-gradient-to-r from-blue-500 to-cyan-500 text-white px-4 py-1">
                                                    Most Popular
                                                </Badge>
                                            </div>
                                        )}
                                        {isEnterprise && (
                                            <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                                                <Badge variant="secondary" className="px-4 py-1">
                                                    Enterprise
                                                </Badge>
                                            </div>
                                        )}

                                        <CardHeader className="text-center pb-4">
                                            <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg bg-gradient-to-r ${getPlanColor()} text-white mb-4 mx-auto`}>
                                                {getPlanIcon(plan.name)}
                                            </div>
                                            <CardTitle className="text-2xl mb-2">{plan.name}</CardTitle>
                                            <CardDescription className="text-base mb-4">
                                                {plan.description || "Perfect for optimizing your AI workflows"}
                                            </CardDescription>

                                            <div className="flex items-baseline justify-center gap-2 mb-4">
                                                <span className="text-4xl font-bold text-foreground">
                                                    {formatPrice(plan.monthly_price_cents)}
                                                </span>
                                                {plan.monthly_price_cents > 0 && (
                                                    <span className="text-muted-foreground">/month</span>
                                                )}
                                                {isContactSalesPlan(plan) && (
                                                    <HelpTooltip content="Custom pricing based on your specific needs and usage requirements." />
                                                )}
                                            </div>

                                            {plan.monthly_price_cents > 0 && (
                                                <p className="text-sm text-muted-foreground">
                                                    Billed monthly - Cancel anytime
                                                </p>
                                            )}
                                        </CardHeader>

                                        <CardContent className="flex-grow">
                                            {/* Key Metrics */}
                                            <div className="grid grid-cols-2 gap-4 mb-6 p-4 bg-muted/30 rounded-lg">
                                                <div className="text-center">
                                                    <div className="text-2xl font-bold text-primary">
                                                        {plan.monthly_quota < 0 ? 'Custom' : plan.monthly_quota.toLocaleString()}
                                                    </div>
                                                    <div className="text-xs text-muted-foreground uppercase tracking-wide">
                                                        Tokens/Month
                                                    </div>
                                                </div>
                                                <div className="text-center">
                                                    <div className="text-2xl font-bold text-primary">
                                                        {plan.max_api_keys}
                                                    </div>
                                                    <div className="text-xs text-muted-foreground uppercase tracking-wide">
                                                        API Keys
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Features */}
                                            <div className="space-y-3">
                                                <h4 className="font-semibold text-sm uppercase tracking-wide text-muted-foreground mb-3">
                                                    What's Included
                                                </h4>
                                                <ul className="space-y-3">
                                                    {plan.features.slice(0, 6).map((feature, i) => (
                                                        <li key={i} className="flex items-start gap-3 text-sm">
                                                            <Check className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                                                            <span className="text-muted-foreground leading-relaxed">{feature}</span>
                                                        </li>
                                                    ))}
                                                    {plan.features.length > 6 && (
                                                        <li className="text-sm text-muted-foreground">
                                                            +{plan.features.length - 6} more features
                                                        </li>
                                                    )}
                                                </ul>
                                            </div>
                                        </CardContent>

                                        <CardFooter className="pt-6">
                                            <Link href={getGettingStartedHref(plan)} className="w-full">
                                                <Button className="h-12 w-full text-base font-semibold" size="lg">
                                                    Getting Started
                                                    <ArrowRight className="ml-2 h-4 w-4" />
                                                </Button>
                                            </Link>
                                            {isContactSalesPlan(plan) && (
                                                <p className="text-xs text-muted-foreground text-center mt-2">
                                                    Custom enterprise pricing
                                                </p>
                                            )}
                                            {!isContactSalesPlan(plan) && plan.monthly_price_cents > 0 && plan.name.toLowerCase().includes('trial') && (
                                                <p className="text-xs text-muted-foreground text-center mt-2">
                                                    14-day free trial - No credit card required
                                                </p>
                                            )}
                                        </CardFooter>
                                    </Card>
                                );
                            })}
                        </div>
                    )}

                    {/* FAQ Section */}
                    <div className="mt-20 max-w-4xl mx-auto">
                        <div className="text-center mb-12">
                            <h2 className="text-3xl font-bold mb-4">Frequently Asked Questions</h2>
                            <p className="text-lg text-muted-foreground">
                                Everything you need to know about our pricing
                            </p>
                        </div>

                        <div className="grid md:grid-cols-2 gap-8">
                            <div className="space-y-6">
                                <div>
                                    <h3 className="font-semibold mb-2">Can I change plans anytime?</h3>
                                    <p className="text-sm text-muted-foreground">
                                        Yes! Upgrade or downgrade your plan at any time. Changes take effect immediately, and billing adjusts accordingly.
                                    </p>
                                </div>
                                <div>
                                    <h3 className="font-semibold mb-2">What happens if I exceed my token limit?</h3>
                                    <p className="text-sm text-muted-foreground">
                                        You'll receive notifications when approaching your limit. Enterprise plans include overage billing. Other plans may have rate limiting until the next billing cycle.
                                    </p>
                                </div>
                            </div>
                            <div className="space-y-6">
                                <div>
                                    <h3 className="font-semibold mb-2">Do you offer annual discounts?</h3>
                                    <p className="text-sm text-muted-foreground">
                                        Yes! Contact our sales team for annual pricing options with significant savings compared to monthly billing.
                                    </p>
                                </div>
                                <div>
                                    <h3 className="font-semibold mb-2">Is there a free trial for paid plans?</h3>
                                    <p className="text-sm text-muted-foreground">
                                        Absolutely! All paid plans include a 14-day free trial. No credit card required to get started.
                                    </p>
                                </div>
                            </div>
                        </div>

                        <div className="text-center mt-12">
                            <Link href="/faq">
                                <Button variant="outline" size="lg">
                                    View All FAQs
                                    <ArrowRight className="ml-2 h-4 w-4" />
                                </Button>
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
        </PublicLayout>
    );
}


