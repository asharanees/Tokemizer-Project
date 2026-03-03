import { useAuth } from "@/contexts/AuthContext";
import { Redirect, Route, RouteProps } from "wouter";
import { Loader2 } from "lucide-react";

export default function AdminRoute(props: RouteProps) {
    const { user, isAuthenticated, isLoading } = useAuth();

    if (isLoading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    if (!isAuthenticated) {
        return <Redirect to="/login" />;
    }

    if (user?.role !== "admin") {
        return <Redirect to="/" />;
    }

    return <Route {...props} />;
}
