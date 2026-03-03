import React, { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { authFetch } from "../lib/authFetch";

interface User {
    id: string;
    email: string;
    name: string;
    phone_number?: string;
    role: "admin" | "customer";
    subscription_status: string;
    subscription_tier: string;
    created_at: string;
    quota_overage_bonus?: number;
}

interface AuthState {
    user: User | null;
    isAuthenticated: boolean;
    isLoading: boolean;
    token: string | null;
    login: (token: string, refresh_token: string, user: User) => void;
    logout: () => void;
    updateUser: (user: User) => void;
}

const AuthContext = createContext<AuthState | undefined>(undefined);

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error("useAuth must be used within an AuthProvider");
    }
    return context;
};

export const AuthProvider = ({ children }: { children: ReactNode }) => {
    const [user, setUser] = useState<User | null>(null);
    const [token, setToken] = useState<string | null>(localStorage.getItem("token"));
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const initAuth = async () => {
            const storedToken = localStorage.getItem("token");
            if (storedToken) {
                setToken(storedToken);
                try {
                    // Verify token and get user details
                    const response = await authFetch("/api/auth/me");

                    if (response.ok) {
                        const userData = await response.json();
                        setUser(userData);
                        setToken(localStorage.getItem("token"));
                    } else {
                        console.warn("Token invalid or expired");
                        logout();
                    }
                } catch (error) {
                    console.error("Auth initialization failed:", error);
                    logout();
                }
            }
            setIsLoading(false);
        };

        initAuth();
    }, []);

    const login = (accessToken: string, refreshToken: string, userData: User) => {
        localStorage.setItem("token", accessToken);
        localStorage.setItem("refresh_token", refreshToken);
        setToken(accessToken);
        setUser(userData);
    };

    const logout = () => {
        localStorage.removeItem("token");
        localStorage.removeItem("refresh_token");
        setToken(null);
        setUser(null);
    };

    const updateUser = (userData: User) => {
        setUser(userData);
    };

    return (
        <AuthContext.Provider
            value={{
                user,
                isAuthenticated: !!user,
                isLoading,
                token,
                login,
                logout,
                updateUser,
            }}
        >
            {children}
        </AuthContext.Provider>
    );
};
