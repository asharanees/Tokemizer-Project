import { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { Loader2 } from "lucide-react";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
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
}

interface RegisterFormProps {
    onLoginClick: () => void;
}

export function RegisterForm({ onLoginClick }: RegisterFormProps) {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [name, setName] = useState("");
    const [selectedPlanId, setSelectedPlanId] = useState("");
    const [plans, setPlans] = useState<Plan[]>([]);
    const [selectedPlan, setSelectedPlan] = useState<Plan | null>(null);
    const { login } = useAuth();
    const [, setLocation] = useLocation();
    const { toast } = useToast();
    const [isLoading, setIsLoading] = useState(false);
    const [isLoadingPlans, setIsLoadingPlans] = useState(true);
    const requestedPlanId = new URLSearchParams(window.location.search).get("plan_id");

    const readErrorMessage = async (response: Response, fallback: string): Promise<string> => {
        const contentType = response.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
            try {
                const payload = await response.json();
                if (payload?.detail && typeof payload.detail === "string") {
                    return payload.detail;
                }
            } catch {
                // Ignore malformed JSON and fall back to plain text.
            }
        }
        const text = (await response.text()).trim();
        if (!text) {
            return fallback;
        }
        if (text.startsWith("<")) {
            return `${fallback} (HTTP ${response.status})`;
        }
        return text;
    };

    useEffect(() => {
        const isSalesPlan = (plan: Plan) =>
            plan.monthly_price_cents < 0 || (plan.annual_price_cents ?? 0) < 0;

        const fetchPlans = async () => {
            try {
                const response = await fetch(toApiUrl("/api/auth/plans"));
                if (!response.ok) throw new Error("Failed to fetch plans");
                const data = await response.json();
                const activePlans = data.filter((p: Plan) => p.is_active);
                setPlans(activePlans);
                if (activePlans.length > 0) {
                    const requestedPlan =
                        requestedPlanId
                            ? activePlans.find((p: Plan) => p.id === requestedPlanId && !isSalesPlan(p))
                            : null;
                    const freePlan = activePlans.find(
                        (p: Plan) => p.monthly_price_cents === 0 && !isSalesPlan(p)
                    );
                    const fallbackPlan = activePlans.find((p: Plan) => !isSalesPlan(p));
                    const initialPlan = requestedPlan || freePlan || fallbackPlan || activePlans[0];
                    setSelectedPlanId(initialPlan.id);
                    setSelectedPlan(initialPlan);
                }
            } catch (error) {
                console.error("Error fetching plans:", error);
                toast({ title: "Error", description: "Failed to load subscription plans", variant: "destructive" });
            } finally {
                setIsLoadingPlans(false);
            }
        };
        fetchPlans();
    }, [requestedPlanId, toast]);

    const handlePlanChange = (planId: string) => {
        setSelectedPlanId(planId);
        const plan = plans.find((p) => p.id === planId);
        if (plan) {
            setSelectedPlan(plan);
        }
    };

    const planRequiresPayment = (plan: Plan) =>
        plan.monthly_price_cents > 0 || (plan.annual_price_cents ?? 0) > 0;

    const planRequiresSalesContact = (plan: Plan) =>
        plan.monthly_price_cents < 0 || (plan.annual_price_cents ?? 0) < 0;

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!selectedPlan) {
            toast({ title: "Error", description: "Please select a plan", variant: "destructive" });
            return;
        }

        setIsLoading(true);

        try {
            const response = await fetch(toApiUrl("/api/auth/register"), {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ email, password, name, plan_id: selectedPlan.id }),
            });

            if (!response.ok) {
                const errorMessage = await readErrorMessage(response, "Registration failed");
                throw new Error(errorMessage);
            }

            await response.json();
            
            // Auto login after registration
            try {
                const formBody = new URLSearchParams();
                formBody.append("username", email);
                formBody.append("password", password);

                const loginResponse = await fetch(toApiUrl("/api/auth/login"), {
                    method: "POST",
                    headers: { "Content-Type": "application/x-www-form-urlencoded" },
                    body: formBody,
                });

                if (loginResponse.ok) {
                    const loginData = await loginResponse.json();
                    login(loginData.access_token, loginData.refresh_token, loginData.user);
                    toast({
                        title: "Account created!",
                        description: "Welcome to Tokemizer.",
                    });
                    setLocation("/");
                } else {
                    // If auto-login fails, redirect to login
                    onLoginClick();
                    toast({
                        title: "Account created!",
                        description: "Please login with your credentials.",
                    });
                }
            } catch (loginError) {
                console.error("Auto-login error:", loginError);
                 onLoginClick();
                 toast({
                    title: "Account created!",
                    description: "Please login with your credentials.",
                });
            }

        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : "Registration failed";
            toast({
                variant: "destructive",
                title: "Registration failed",
                description: message,
            });
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="w-full">
            <CardHeader>
                <CardTitle>Create an Account</CardTitle>
                <CardDescription>Join Tokemizer to start optimizing your prompts</CardDescription>
            </CardHeader>
            <form onSubmit={handleSubmit}>
                <CardContent className="space-y-4">
                    <div className="space-y-2">
                        <Label htmlFor="name">Full Name</Label>
                        <Input
                            id="name"
                            placeholder="John Doe"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            required
                        />
                    </div>
                    <div className="space-y-2">
                        <Label htmlFor="email">Email</Label>
                        <Input
                            id="email"
                            type="email"
                            placeholder="name@example.com"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                        />
                    </div>
                    <div className="space-y-2">
                        <Label htmlFor="password">Password</Label>
                        <Input
                            id="password"
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>
                    
                    <div className="space-y-2">
                        <Label>Subscription Plan</Label>
                        {isLoadingPlans ? (
                             <div className="h-10 w-full animate-pulse rounded-md bg-muted" />
                        ) : (
                            <Select value={selectedPlanId} onValueChange={handlePlanChange}>
                                <SelectTrigger>
                                    <SelectValue placeholder="Select a plan" />
                                </SelectTrigger>
                                <SelectContent>
                                    {plans.map((plan) => (
                                        <SelectItem key={plan.id} value={plan.id} disabled={planRequiresSalesContact(plan)}>
                                            <div className="flex items-center gap-2">
                                                <span className="font-medium">{plan.name}</span>
                                                {planRequiresSalesContact(plan) ? (
                                                    <Badge variant="outline" className="text-xs">Contact Sales</Badge>
                                                ) : plan.monthly_price_cents === 0 ? (
                                                     <Badge variant="secondary" className="text-xs">Free</Badge>
                                                ) : (
                                                    <span className="text-muted-foreground text-xs">
                                                        ${(plan.monthly_price_cents / 100).toFixed(2)}/mo
                                                    </span>
                                                )}
                                            </div>
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        )}
                        {selectedPlan && (
                            <p className="text-xs text-muted-foreground mt-1">
                                {selectedPlan.description || "Includes standard features."}
                                {planRequiresSalesContact(selectedPlan)
                                    ? " (Contact sales required)"
                                    : planRequiresPayment(selectedPlan)
                                        ? " (Payment method required later)"
                                        : ""}
                            </p>
                        )}
                    </div>
                </CardContent>
                <CardFooter className="flex flex-col space-y-4">
                    <Button type="submit" className="w-full" disabled={isLoading || isLoadingPlans}>
                        {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : "Create Account"}
                    </Button>
                    <div className="text-sm text-center text-muted-foreground">
                        Already have an account?{" "}
                        <button
                            type="button"
                            onClick={onLoginClick}
                            className="text-primary hover:underline font-medium"
                        >
                            Login
                        </button>
                    </div>
                </CardFooter>
            </form>
        </div>
    );
}
