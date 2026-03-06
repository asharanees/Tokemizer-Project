import { useEffect, useMemo, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { Loader2 } from "lucide-react";
import { toApiUrl } from "@/lib/apiUrl";

interface LoginFormProps {
    onRegisterClick: () => void;
}

const directEc2ControlBaseUrl = (import.meta.env.VITE_EC2_CONTROL_URL || "")
    .trim()
    .replace(/\/+$/, "");

export function LoginForm({ onRegisterClick }: LoginFormProps) {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const { login } = useAuth();
    const [, setLocation] = useLocation();
    const { toast } = useToast();
    const [isLoading, setIsLoading] = useState(false);
    const [ec2Status, setEc2Status] = useState<string>("unknown");
    const [isInfraBusy, setIsInfraBusy] = useState(false);
    const [infraError, setInfraError] = useState<string>("");

    const ec2StatusLabel = useMemo(() => {
        const normalized = ec2Status.toLowerCase();
        if (!normalized || normalized === "unknown") {
            return "Unknown";
        }
        return normalized.charAt(0).toUpperCase() + normalized.slice(1);
    }, [ec2Status]);

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
        // Avoid showing full HTML error pages in toast messages.
        if (text.startsWith("<")) {
            return `${fallback} (HTTP ${response.status})`;
        }
        return text;
    };

    const requestEc2Control = async (action: "start" | "stop" | "status") => {
        const backendPath =
            action === "status"
                ? "/api/auth/infrastructure/ec2/status"
                : `/api/auth/infrastructure/ec2/${action}`;
        const backendMethod = action === "status" ? "GET" : "POST";
        try {
            const response = await fetch(toApiUrl(backendPath), {
                method: backendMethod,
            });
            if (!response.ok) {
                const message = await readErrorMessage(response, `Unable to ${action} EC2`);
                throw new Error(message);
            }
            return await response.json();
        } catch (primaryError: unknown) {
            if (!directEc2ControlBaseUrl) {
                throw primaryError;
            }
            const directUrl = `${directEc2ControlBaseUrl}/?action=${action}`;
            const fallbackResponse = await fetch(directUrl, {
                method: "GET",
            });
            if (!fallbackResponse.ok) {
                const message = await readErrorMessage(fallbackResponse, `Unable to ${action} EC2`);
                throw new Error(message);
            }
            return await fallbackResponse.json();
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);

        try {
            const formBody = new URLSearchParams();
            formBody.append("username", email);
            formBody.append("password", password);

            const response = await fetch(toApiUrl("/api/auth/login"), {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                body: formBody,
            });

            if (!response.ok) {
                const errorMessage = await readErrorMessage(response, "Login failed");
                throw new Error(errorMessage);
            }

            const data = await response.json();
            login(data.access_token, data.refresh_token, data.user);
            toast({
                title: "Welcome back!",
                description: "You have successfully logged in.",
            });
            setLocation("/");
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : "Login failed";
            toast({
                variant: "destructive",
                title: "Login failed",
                description: message,
            });
        } finally {
            setIsLoading(false);
        }
    };

    const fetchEc2Status = async () => {
        setIsInfraBusy(true);
        try {
            const data = await requestEc2Control("status");
            const instanceState =
                typeof data?.instance_state === "string"
                    ? data.instance_state
                    : typeof data?.details?.state === "string"
                      ? data.details.state
                      : "unknown";
            setEc2Status(instanceState);
            setInfraError("");
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : "Unable to fetch EC2 status";
            setInfraError(message);
            toast({
                variant: "destructive",
                title: "EC2 status error",
                description: message,
            });
        } finally {
            setIsInfraBusy(false);
        }
    };

    const runEc2Action = async (action: "start" | "stop") => {
        setIsInfraBusy(true);
        try {
            const data = await requestEc2Control(action);
            const nextState =
                typeof data?.instance_state === "string"
                    ? data.instance_state
                    : action === "start"
                      ? "pending"
                      : "stopping";
            setEc2Status(nextState);
            setInfraError("");
            toast({
                title: `EC2 ${action} requested`,
                description: `Current state: ${nextState}`,
            });
            window.setTimeout(() => {
                void fetchEc2Status();
            }, 1500);
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : `Unable to ${action} EC2`;
            setInfraError(message);
            toast({
                variant: "destructive",
                title: `EC2 ${action} failed`,
                description: message,
            });
        } finally {
            setIsInfraBusy(false);
        }
    };

    useEffect(() => {
        void fetchEc2Status();
    }, []);

    return (
        <div className="w-full">
            <CardHeader>
                <CardTitle>Login to Tokemizer</CardTitle>
                <CardDescription>Enter your email and password to access your account</CardDescription>
            </CardHeader>
            <form onSubmit={handleSubmit}>
                <CardContent className="space-y-4">
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
                </CardContent>
                <CardFooter className="flex flex-col space-y-4">
                    <Button type="submit" className="w-full" disabled={isLoading}>
                        {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : "Login"}
                    </Button>
                    <div className="text-sm text-center text-muted-foreground">
                        Don't have an account?{" "}
                        <button
                            type="button"
                            onClick={onRegisterClick}
                            className="text-primary hover:underline font-medium"
                        >
                            Register
                        </button>
                    </div>
                    <div className="w-full rounded-md border border-border p-3 space-y-3">
                        <div className="text-sm font-medium">Server Control</div>
                        <div className="text-xs text-muted-foreground">
                            EC2 status: <span className="font-medium">{ec2StatusLabel}</span>
                        </div>
                        {infraError && <div className="text-xs text-destructive">{infraError}</div>}
                        <div className="grid grid-cols-3 gap-2">
                            <Button
                                type="button"
                                variant="outline"
                                onClick={() => {
                                    void runEc2Action("start");
                                }}
                                disabled={isInfraBusy}
                            >
                                Start
                            </Button>
                            <Button
                                type="button"
                                variant="outline"
                                onClick={() => {
                                    void runEc2Action("stop");
                                }}
                                disabled={isInfraBusy}
                            >
                                Stop
                            </Button>
                            <Button
                                type="button"
                                variant="outline"
                                onClick={() => {
                                    void fetchEc2Status();
                                }}
                                disabled={isInfraBusy}
                            >
                                {isInfraBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : "Status"}
                            </Button>
                        </div>
                    </div>
                </CardFooter>
            </form>
        </div>
    );
}
