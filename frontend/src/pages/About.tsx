import { PublicLayout } from "@/components/layout/PublicLayout";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Check, Cpu, Lock, Zap, BarChart3, Globe, Shield, Clock, Users, TrendingDown } from "lucide-react";
import { Link } from "wouter";
import { seoDefaults, useSeo } from "@/lib/seo";

export default function About() {
  useSeo({
    title: "About Tokemizer | Enterprise Prompt Optimization Platform",
    description:
      "Learn how Tokemizer helps AI engineering teams reduce token costs by 70% and optimize LLM latency with secure, enterprise-grade semantic compression.",
    path: "/about",
    structuredData: {
      "@context": "https://schema.org",
      "@type": "Organization",
      name: seoDefaults.siteName,
      url: seoDefaults.siteUrl,
      description:
        "Tokemizer is an enterprise platform for optimizing prompt tokens and reducing AI inference costs with 30-70% compression rates.",
    },
  });

  return (
    <PublicLayout>
        <div className="bg-background">
            {/* Hero Section */}
            <div className="relative overflow-hidden bg-gradient-to-br from-primary/5 via-background to-secondary/5">
                <div className="container mx-auto px-4 py-16 lg:py-24">
                    <div className="text-center mb-16">
                        <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-2 rounded-full text-sm font-medium mb-6">
                            <Zap className="h-4 w-4" />
                            Enterprise-Grade AI Token Optimizer
                        </div>
                        <h1 className="text-4xl font-extrabold tracking-tight lg:text-7xl mb-6 text-foreground">
                            Cut LLM Costs upto
                            <span className="text-primary"> 70%</span>
                        </h1>
                        <p className="text-xl text-muted-foreground max-w-3xl mx-auto mb-8 leading-relaxed">
                            Tokemizer intelligently compresses your AI prompts while preserving meaning, context, and intent.
                            Deploy anywhere, including cloud, on-prem, or air-gapped environments, with millisecond performance and enterprise security.
                        </p>
                        <div className="flex flex-col sm:flex-row justify-center gap-4 mb-12">
                            <Link href="/login?action=register">
                                 <Button size="lg" className="h-14 px-8 text-lg font-semibold">
                                    Start Saving Today
                                    <TrendingDown className="ml-2 h-5 w-5" />
                                </Button>
                            </Link>
                            <Link href="/plans">
                                <Button variant="outline" size="lg" className="h-14 px-8 text-lg">
                                    View Pricing
                                </Button>
                            </Link>
                        </div>

                        {/* Key Stats */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto">
                            <div className="text-center">
                                <div className="text-3xl font-bold text-primary mb-2">70%</div>
                                <div className="text-sm text-muted-foreground">Upto 70% Token Savings</div>
                            </div>
                            <div className="text-center">
                                <div className="text-3xl font-bold text-primary mb-2">&lt;100ms</div>
                                <div className="text-sm text-muted-foreground">Response Time</div>
                            </div>
                            <div className="text-center">
                                <div className="text-3xl font-bold text-primary mb-2">500K+</div>
                                <div className="text-sm text-muted-foreground">Tokens Supported</div>
                            </div>
                            <div className="text-center">
                                <div className="text-3xl font-bold text-primary mb-2">99.9%</div>
                                <div className="text-sm text-muted-foreground">Uptime SLA</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Features Grid */}
            <div className="container mx-auto px-4 py-16 lg:py-24">
                <div className="text-center mb-16">
                    <h2 className="text-3xl font-bold mb-4">Why Choose Tokemizer?</h2>
                    <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                        Built for enterprises that demand performance, security, and reliability
                    </p>
                </div>

                <div className="grid md:grid-cols-3 gap-8 mb-20">
                    <Card className="border-border/40 shadow-sm bg-card/50 backdrop-blur hover:shadow-lg transition-shadow">
                        <CardHeader>
                            <div className="mb-4 h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center">
                                <BarChart3 className="h-6 w-6 text-primary" />
                            </div>
                            <CardTitle>Intelligent Compression</CardTitle>
                            <CardDescription className="text-base mt-2">
                                Advanced rule-based optimization with three modes: Conservative (30-35% savings), Balanced (40-45% savings), and Maximum (50-70% savings). Automatically preserves code, URLs, numbers, and structured data.
                            </CardDescription>
                        </CardHeader>
                    </Card>
                    <Card className="border-border/40 shadow-sm bg-card/50 backdrop-blur hover:shadow-lg transition-shadow">
                        <CardHeader>
                            <div className="mb-4 h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center">
                                <Clock className="h-6 w-6 text-primary" />
                            </div>
                            <CardTitle>Lightning Fast</CardTitle>
                            <CardDescription className="text-base mt-2">
                                Sub-millisecond optimization for most prompts with intelligent chunking for large documents up to 500K+ tokens. Memory-safe processing prevents OOM errors on ultra-long content.
                            </CardDescription>
                        </CardHeader>
                    </Card>
                    <Card className="border-border/40 shadow-sm bg-card/50 backdrop-blur hover:shadow-lg transition-shadow">
                        <CardHeader>
                            <div className="mb-4 h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center">
                                <Shield className="h-6 w-6 text-primary" />
                            </div>
                            <CardTitle>Enterprise Security</CardTitle>
                            <CardDescription className="text-base mt-2">
                                SOC 2 compliant with TLS 1.3 encryption, AES-256 data protection, and configurable data handling. Deploy on-prem or air-gapped with full data residency control.
                            </CardDescription>
                        </CardHeader>
                    </Card>
                </div>

                {/* Use Cases */}
                <section className="mb-20">
                    <div className="text-center mb-12">
                        <h2 className="text-3xl font-bold mb-4">Perfect For Every AI Workflow</h2>
                        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                            From chatbots to enterprise RAG systems, Tokemizer optimizes any LLM application
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                        <div className="bg-card border rounded-lg p-6 hover:shadow-md transition-shadow">
                            <div className="h-10 w-10 rounded-lg bg-blue-500/10 flex items-center justify-center mb-4">
                                <Users className="h-5 w-5 text-blue-500" />
                            </div>
                            <h3 className="font-semibold mb-2">Customer Support Chatbots</h3>
                            <p className="text-muted-foreground text-sm">
                                Compress conversation history and system prompts to handle longer customer interactions within token limits.
                            </p>
                        </div>
                        <div className="bg-card border rounded-lg p-6 hover:shadow-md transition-shadow">
                            <div className="h-10 w-10 rounded-lg bg-green-500/10 flex items-center justify-center mb-4">
                                <Globe className="h-5 w-5 text-green-500" />
                            </div>
                            <h3 className="font-semibold mb-2">RAG & Document Processing</h3>
                            <p className="text-muted-foreground text-sm">
                                Optimize retrieved documents and context windows to maximize relevance while staying within model constraints.
                            </p>
                        </div>
                        <div className="bg-card border rounded-lg p-6 hover:shadow-md transition-shadow">
                            <div className="h-10 w-10 rounded-lg bg-purple-500/10 flex items-center justify-center mb-4">
                                <BarChart3 className="h-5 w-5 text-purple-500" />
                            </div>
                            <h3 className="font-semibold mb-2">Batch Processing</h3>
                            <p className="text-muted-foreground text-sm">
                                Process thousands of prompts efficiently with synchronous batch optimization and detailed per-item metrics.
                            </p>
                        </div>
                        <div className="bg-card border rounded-lg p-6 hover:shadow-md transition-shadow">
                            <div className="h-10 w-10 rounded-lg bg-orange-500/10 flex items-center justify-center mb-4">
                                <Cpu className="h-5 w-5 text-orange-500" />
                            </div>
                            <h3 className="font-semibold mb-2">API Middleware</h3>
                            <p className="text-muted-foreground text-sm">
                                Drop into existing LLM pipelines as transparent middleware to reduce costs without changing application code.
                            </p>
                        </div>
                        <div className="bg-card border rounded-lg p-6 hover:shadow-md transition-shadow">
                            <div className="h-10 w-10 rounded-lg bg-red-500/10 flex items-center justify-center mb-4">
                                <Lock className="h-5 w-5 text-red-500" />
                            </div>
                            <h3 className="font-semibold mb-2">Enterprise Compliance</h3>
                            <p className="text-muted-foreground text-sm">
                                Maintain data residency with on-prem deployment while ensuring compliance with industry regulations.
                            </p>
                        </div>
                        <div className="bg-card border rounded-lg p-6 hover:shadow-md transition-shadow">
                            <div className="h-10 w-10 rounded-lg bg-cyan-500/10 flex items-center justify-center mb-4">
                                <TrendingDown className="h-5 w-5 text-cyan-500" />
                            </div>
                            <h3 className="font-semibold mb-2">Cost Optimization</h3>
                            <p className="text-muted-foreground text-sm">
                                Reduce LLM API costs upto 70% through intelligent compression without sacrificing output quality.
                            </p>
                        </div>
                    </div>
                </section>

                {/* Production Ready Section */}
                <section className="bg-muted/30 rounded-3xl p-8 lg:p-16 mb-20">
                    <div className="grid lg:grid-cols-2 gap-12 items-center">
                        <div>
                            <h2 className="text-3xl font-bold mb-6">Built for Enterprise Scale</h2>
                            <p className="text-lg text-muted-foreground leading-relaxed mb-6">
                                Tokemizer runs as Dockerized services with a React UI and FastAPI backend, making deployment predictable in any environment.
                                It is designed for local processing, deterministic behavior, and configurable controls that fit enterprise requirements.
                            </p>
                            <p className="text-lg text-muted-foreground leading-relaxed mb-8">
                                From single-tenant on-prem installs to horizontally scaled clusters, the architecture supports high-throughput optimization without compromising data residency.
                            </p>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="flex items-center gap-3">
                                    <Check className="h-5 w-5 text-green-500 shrink-0" />
                                    <span className="text-sm">Docker & Kubernetes Ready</span>
                                </div>
                                <div className="flex items-center gap-3">
                                    <Check className="h-5 w-5 text-green-500 shrink-0" />
                                    <span className="text-sm">Air-Gapped Deployment</span>
                                </div>
                                <div className="flex items-center gap-3">
                                    <Check className="h-5 w-5 text-green-500 shrink-0" />
                                    <span className="text-sm">Horizontal Scaling</span>
                                </div>
                                <div className="flex items-center gap-3">
                                    <Check className="h-5 w-5 text-green-500 shrink-0" />
                                    <span className="text-sm">99.9% Uptime SLA</span>
                                </div>
                            </div>
                        </div>
                        <div className="space-y-6">
                            <div className="bg-card border rounded-lg p-6">
                                <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                                    <Shield className="h-5 w-5 text-primary" />
                                    Security First
                                </h3>
                                <p className="text-muted-foreground text-sm">
                                    SOC 2 compliant with end-to-end encryption, audit logging, and configurable data retention policies.
                                </p>
                            </div>
                            <div className="bg-card border rounded-lg p-6">
                                <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                                    <Globe className="h-5 w-5 text-primary" />
                                    Model Agnostic
                                </h3>
                                <p className="text-muted-foreground text-sm">
                                    Works seamlessly with OpenAI, Anthropic, Google, and open-source models like Llama and Mistral.
                                </p>
                            </div>
                            <div className="bg-card border rounded-lg p-6">
                                <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                                    <BarChart3 className="h-5 w-5 text-primary" />
                                    Full Observability
                                </h3>
                                <p className="text-muted-foreground text-sm">
                                    Comprehensive telemetry, performance metrics, and optimization analytics for complete visibility.
                                </p>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Final CTA */}
                <section className="text-center bg-gradient-to-r from-primary/5 to-secondary/5 rounded-3xl p-12">
                    <h2 className="text-3xl font-bold mb-4">Ready to Optimize Your AI Costs?</h2>
                    <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
                        Join thousands of teams already saving money and improving performance with Tokemizer.
                    </p>
                    <div className="flex flex-col sm:flex-row justify-center gap-4">
                        <Link href="/login?action=register">
                            <Button size="lg" className="h-12 px-8 text-lg">
                                Start Free Trial
                            </Button>
                        </Link>
                        <Link href="/contact">
                            <Button variant="outline" size="lg" className="h-12 px-8 text-lg">
                                Contact Sales
                            </Button>
                        </Link>
                    </div>
                    <p className="text-sm text-muted-foreground mt-4">
                        No credit card required • 14-day free trial • Cancel anytime
                    </p>
                </section>
            </div>
        </div>
    </PublicLayout>
  );
}
