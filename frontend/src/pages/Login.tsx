import { useState } from "react";
import { PublicLayout } from "@/components/layout/PublicLayout";
import { LoginForm } from "@/components/auth/LoginForm";
import { RegisterForm } from "@/components/auth/RegisterForm";
import { Card } from "@/components/ui/card";
import { Zap, Shield, BarChart3, Globe } from "lucide-react";
import { useSeo } from "@/lib/seo";

export default function Login() {
    useSeo({
        title: "Login - Tokemizer | Enterprise Prompt Token Optimizer",
        description: "Login to Tokemizer | Enterprise Prompt Token Optimizer. Securely manage API keys, monitor savings, and configure optimization rules.",
        path: "/login",
    });

    const [isLogin, setIsLogin] = useState(() => {
        const params = new URLSearchParams(window.location.search);
        return params.get("action") !== "register";
    });

    const toggleAuthMode = () => {
        setIsLogin(!isLogin);
    };

    return (
        <PublicLayout>
            <div className="relative isolate overflow-hidden bg-background">
                {/* Background decorative elements could go here */}

                <div className="mx-auto max-w-7xl px-6 pb-24 pt-10 sm:pb-32 lg:flex lg:px-8 lg:py-40">
                    <div className="mx-auto max-w-2xl lg:mx-0 lg:max-w-xl lg:flex-shrink-0 lg:pt-8">
                        <div className="mt-24 sm:mt-32 lg:mt-16">
                            <a href="#" className="inline-flex space-x-6">
                                <span className="rounded-full bg-primary/10 px-3 py-1 text-sm font-semibold leading-6 text-primary ring-1 ring-inset ring-primary/10">
                                    What's new
                                </span>
                                <span className="inline-flex items-center space-x-2 text-sm font-medium leading-6 text-muted-foreground">
                                    <span>Just shipped v2.0</span>
                                </span>
                            </a>
                        </div>
                        <h1 className="mt-10 text-4xl font-bold tracking-tight text-foreground sm:text-6xl">
                            Optimize LLM Token Usage with Enterprise Precision
                        </h1>
                        <p className="mt-6 text-lg leading-8 text-muted-foreground">
                            Tokemizer is a lightweight, robust, and configurable prompt token reduction engine with millisecond response times.
                            Deploy on-prem with full air-gapped support and scale from single-node installs to multi-instance clusters.
                        </p>
                        <div className="mt-10 flex items-center gap-x-6">
                            <div className="flex flex-wrap gap-4">
                                <div className="flex flex-col gap-1">
                                    <div className="flex items-center gap-2 font-semibold">
                                        <Zap className="h-5 w-5 text-yellow-500" />
                                        <span>Reduce Costs</span>
                                    </div>
                                    <p className="text-sm text-muted-foreground">Up to 70% token savings</p>
                                </div>
                                <div className="flex flex-col gap-1">
                                    <div className="flex items-center gap-2 font-semibold">
                                        <Shield className="h-5 w-5 text-blue-500" />
                                        <span>Enterprise Security</span>
                                    </div>
                                    <p className="text-sm text-muted-foreground">SOC2 Compliant Ready</p>
                                </div>
                                <div className="flex flex-col gap-1">
                                    <div className="flex items-center gap-2 font-semibold">
                                        <Globe className="h-5 w-5 text-emerald-500" />
                                        <span>On-Prem Ready</span>
                                    </div>
                                    <p className="text-sm text-muted-foreground">Air-gapped deployment support</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="mx-auto mt-16 flex max-w-2xl sm:mt-24 lg:ml-10 lg:mt-0 lg:mr-0 lg:max-w-none lg:flex-none xl:ml-32">
                        <div className="max-w-md flex-none sm:max-w-5xl lg:max-w-none">
                            <Card className="w-full max-w-md shadow-2xl border-primary/20 bg-card/95 backdrop-blur">
                                {isLogin ? (
                                    <LoginForm onRegisterClick={toggleAuthMode} />
                                ) : (
                                    <RegisterForm onLoginClick={toggleAuthMode} />
                                )}
                            </Card>
                        </div>
                    </div>
                </div>

                {/* Features Section */}
                <div className="mx-auto max-w-7xl px-6 lg:px-8 py-24 sm:py-32 bg-muted/30">
                    <div className="mx-auto max-w-2xl lg:text-center">
                        <h2 className="text-base font-semibold leading-7 text-primary">Deploy Faster</h2>
                        <p className="mt-2 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
                            Everything you need to scale AI
                        </p>
                        <p className="mt-6 text-lg leading-8 text-muted-foreground">
                            Our platform provides the essential infrastructure to optimize, monitor, and control your LLM traffic.
                        </p>
                    </div>
                    <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-4xl">
                        <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-10 lg:max-w-none lg:grid-cols-2 lg:gap-y-16">
                            <div className="relative pl-16">
                                <dt className="text-base font-semibold leading-7 text-foreground">
                                    <div className="absolute left-0 top-0 flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                                        <Zap className="h-6 w-6 text-primary-foreground" aria-hidden="true" />
                                    </div>
                                    Smart Compression
                                </dt>
                                <dd className="mt-2 text-base leading-7 text-muted-foreground">
                                    Deterministic, rule-based compression with three optimization modes to preserve intent and context.
                                </dd>
                            </div>
                            <div className="relative pl-16">
                                <dt className="text-base font-semibold leading-7 text-foreground">
                                    <div className="absolute left-0 top-0 flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                                        <Shield className="h-6 w-6 text-primary-foreground" aria-hidden="true" />
                                    </div>
                                    PII Protection
                                </dt>
                                <dd className="mt-2 text-base leading-7 text-muted-foreground">
                                    Content protection preserves sensitive patterns, code, and structure for compliance.
                                </dd>
                            </div>
                            <div className="relative pl-16">
                                <dt className="text-base font-semibold leading-7 text-foreground">
                                    <div className="absolute left-0 top-0 flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                                        <BarChart3 className="h-6 w-6 text-primary-foreground" aria-hidden="true" />
                                    </div>
                                    Analytics & Insights
                                </dt>
                                <dd className="mt-2 text-base leading-7 text-muted-foreground">
                                    Track token savings, per-pass performance, and optimization trends in real time.
                                </dd>
                            </div>
                            <div className="relative pl-16">
                                <dt className="text-base font-semibold leading-7 text-foreground">
                                    <div className="absolute left-0 top-0 flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                                        <Globe className="h-6 w-6 text-primary-foreground" aria-hidden="true" />
                                    </div>
                                    Universal Compatibility
                                </dt>
                                <dd className="mt-2 text-base leading-7 text-muted-foreground">
                                    Model-agnostic API with batch and streaming endpoints for any LLM stack.
                                </dd>
                            </div>
                        </dl>
                    </div>
                </div>
            </div>
        </PublicLayout>
    );
}
