import { useState } from "react";
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

export function LoginForm({ onRegisterClick }: LoginFormProps) {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const { login } = useAuth();
    const [, setLocation] = useLocation();
    const { toast } = useToast();
    const [isLoading, setIsLoading] = useState(false);

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
                </CardFooter>
            </form>
        </div>
    );
}
