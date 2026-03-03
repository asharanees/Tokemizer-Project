import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { PublicLayout } from "@/components/layout/PublicLayout";
import { Button } from "@/components/ui/button";
import { Link } from "wouter";
import { HelpCircle, TrendingDown, Shield, Zap, Users, Globe } from "lucide-react";
import { useSeo } from "@/lib/seo";

const faqs = [
  {
    category: "Getting Started",
    icon: <Zap className="h-5 w-5" />,
    items: [
      {
        q: "What is Tokemizer and how does it help me?",
        a: "Tokemizer is an enterprise-grade AI token optimization platform that reduces Large Language Model (LLM) token usage upto 70% while preserving semantic meaning, context, and intent. It helps you cut AI costs dramatically without sacrificing output quality, perfect for chatbots, RAG systems, batch processing, and any LLM application."
      },
      {
        q: "Who is Tokemizer built for?",
        a: "We built Tokemizer for AI engineers, developers, platform teams, and enterprises scaling LLM applications. Whether you're a startup building a chatbot, a data science team processing documents, or an enterprise with millions of daily requests, Tokemizer helps you control costs and improve performance."
      },
      {
        q: "How do I get started?",
        a: "Getting started is simple: sign up for a free account, grab your API key, and start optimizing immediately. Our web Playground lets you test optimizations manually, or integrate our API directly into your application code. No credit card required for the free tier."
      },
      {
        q: "Do you have a free trial?",
        a: "Yes! Our Free Tier gives you generous monthly token allowances to test the platform. All paid plans also include a 14-day free trial with full access to premium features. No credit card required to get started."
      }
    ]
  },
  {
    category: "Technology & Optimization",
    icon: <TrendingDown className="h-5 w-5" />,
    items: [
      {
        q: "How does the optimization actually work?",
        a: "Our proprietary engine uses advanced rule-based NLP and semantic analysis to compress prompts intelligently. It removes redundancies, normalizes terminology, preserves critical content (code, URLs, numbers), and applies domain-specific optimizations. We offer three modes: Conservative (30-35% savings), Balanced (40-45% savings, recommended), and Maximum (50-70% savings) to match your risk tolerance."
      },
      {
        q: "Will optimizing prompts affect the accuracy of LLM responses?",
        a: "We prioritize semantic integrity above all else. Our optimization preserves meaning, context, and intent through intelligent content protection and semantic guardrails. In 99% of cases, Balanced mode produces identical or better results than the original prompt. We include comprehensive validation and can roll back any optimization that doesn't meet quality standards."
      },
      {
        q: "Which LLMs do you support?",
        a: "Tokemizer is model-agnostic and works seamlessly with all major LLMs including OpenAI (GPT-4, GPT-3.5), Anthropic (Claude), Google (Gemini), and open-source models like Llama 3, Mistral, and others. Our optimization principles apply universally across transformer-based models."
      },
      {
        q: "How fast is the optimization process?",
        a: "Lightning fast! Most prompts are optimized in under 100ms, with support for large documents up to 500K+ tokens through intelligent chunking. We use memory-safe processing to prevent OOM errors and maintain sub-second performance even for complex optimizations."
      }
    ]
  },
  {
    category: "Pricing & Billing",
    icon: <HelpCircle className="h-5 w-5" />,
    items: [
      {
        q: "How does your pricing work?",
        a: "We offer flexible, usage-based pricing with three tiers: Free (generous limits for testing), Pro (unlimited usage for growing teams), and Enterprise (custom pricing for large organizations). You're billed based on input tokens processed, with transparent overage protection and no hidden fees."
      },
      {
        q: "Can I change plans anytime?",
        a: "Absolutely! Upgrade or downgrade your plan at any time with changes taking effect immediately. Billing adjusts proportionally, and we prorate all charges. No long-term contracts or cancellation fees."
      },
      {
        q: "What happens if I exceed my plan limits?",
        a: "We notify you when approaching your limit. Enterprise customers get soft limits with overage billing. Other plans may experience rate limiting until the next billing cycle. You can upgrade instantly through your dashboard to continue uninterrupted service."
      },
      {
        q: "Do you offer annual discounts?",
        a: "Yes! Contact our sales team for annual pricing options with significant savings (typically 20-30% off) compared to monthly billing. Annual plans also include priority support and additional features."
      }
    ]
  },
  {
    category: "Security & Compliance",
    icon: <Shield className="h-5 w-5" />,
    items: [
      {
        q: "Is my prompt data secure?",
        a: "Absolutely. Security is our top priority, and we're SOC 2 Type II compliant with TLS 1.3 encryption for data in transit and AES-256 encryption at rest. We use industry-standard security practices and never use customer data to train our models."
      },
      {
        q: "Do you use my data to train your models?",
        a: "No. We strictly do not use customer data to train our foundational models. Your prompts are processed solely for optimization and are not retained longer than necessary for debugging (unless you opt-in to history retention for analytics)."
      },
      {
        q: "Can I deploy Tokemizer on-premise?",
        a: "Yes! For enterprises with strict data residency requirements, we offer self-hosted and air-gapped deployment options via Docker containers or Kubernetes Helm charts. Your data never leaves your infrastructure."
      },
      {
        q: "What about data residency and compliance?",
        a: "We support multiple deployment models to meet your compliance needs: cloud-hosted (SOC 2 compliant), on-premise (full data control), or air-gapped (no external connectivity). All deployments maintain the same security standards and feature parity."
      }
    ]
  },
  {
    category: "Integration & Support",
    icon: <Users className="h-5 w-5" />,
    items: [
      {
        q: "How do I integrate Tokemizer into my application?",
        a: "Integration is straightforward with our simple REST API. Send your prompts to our /api/v1/optimize endpoint and receive optimized versions instantly. We provide SDKs for Python, JavaScript, and Go, plus middleware for popular frameworks. Most integrations take under an hour."
      },
      {
        q: "What kind of support do you provide?",
        a: "All plans include comprehensive documentation, community forums, and email support. Pro plans add priority support with faster response times. Enterprise customers get dedicated success managers, custom integrations, and phone support."
      },
      {
        q: "Can I customize the optimization for my domain?",
        a: "Yes! We support custom canonical mappings for domain-specific terminology, configurable preservation rules, and custom optimization profiles. Enterprise customers can work with our team to build specialized optimization pipelines for their use cases."
      },
      {
        q: "Do you offer training or consulting services?",
        a: "Yes, for Enterprise customers we provide implementation consulting, custom model training, and optimization strategy sessions. Our experts can help you maximize cost savings and performance improvements for your specific use cases."
      }
    ]
  }
];

export default function FAQ() {
  useSeo({
    title: "FAQ",
    description:
      "Common questions about Tokemizer's token classification, security, integration with OpenAI/Anthropic, and pricing models.",
    path: "/faq",
    structuredData: {
      "@context": "https://schema.org",
      "@type": "FAQPage",
      mainEntity: faqs.flatMap((group) =>
        group.items.map((item) => ({
          "@type": "Question",
          name: item.q,
          acceptedAnswer: {
            "@type": "Answer",
            text: item.a,
          },
        })),
      ),
    },
  });

  return (
    <PublicLayout>
      <div className="container mx-auto px-4 py-20 max-w-6xl">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-2 rounded-full text-sm font-medium mb-6">
            <HelpCircle className="h-4 w-4" />
            Knowledge Base
          </div>
          <h1 className="text-4xl font-bold mb-4">Frequently Asked Questions</h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Everything you need to know about optimizing your AI costs and workflow with Tokemizer.
            Can't find what you're looking for? Our support team is here to help.
          </p>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-16 max-w-4xl mx-auto">
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

        <div className="space-y-16">
          {faqs.map((group) => (
            <div key={group.category} className="scroll-mt-20" id={group.category.toLowerCase().replace(/\s+/g, '-')}>
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-primary/10 rounded-lg text-primary">
                  {group.icon}
                </div>
                <h2 className="text-2xl font-semibold text-primary">
                  {group.category}
                </h2>
              </div>
              <Accordion type="single" collapsible className="w-full">
                {group.items.map((item, i) => (
                  <AccordionItem key={i} value={`item-${group.category}-${i}`} className="border-b-border/50">
                    <AccordionTrigger className="text-left font-medium text-lg hover:text-primary transition-colors py-4">
                      {item.q}
                    </AccordionTrigger>
                    <AccordionContent className="text-muted-foreground leading-relaxed text-base pb-6">
                      {item.a}
                    </AccordionContent>
                  </AccordionItem>
                ))}
              </Accordion>
            </div>
          ))}
        </div>

        {/* Help Section */}
        <div className="mt-20 grid md:grid-cols-2 gap-8">
          <div className="bg-card border rounded-2xl p-8 shadow-sm">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <Users className="h-5 w-5 text-blue-500" />
              </div>
              <h3 className="text-xl font-semibold">Developer Community</h3>
            </div>
            <p className="text-muted-foreground mb-6">
              Join thousands of developers optimizing their AI workflows. Get help from the community, share best practices, and learn from real-world implementations.
            </p>
            <Button variant="outline" className="w-full">
              Join Community
            </Button>
          </div>

          <div className="bg-card border rounded-2xl p-8 shadow-sm">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-green-500/10 rounded-lg">
                <Globe className="h-5 w-5 text-green-500" />
              </div>
              <h3 className="text-xl font-semibold">Technical Documentation</h3>
            </div>
            <p className="text-muted-foreground mb-6">
              Comprehensive API docs, integration guides, and best practices. From quick starts to advanced configurations, find everything you need to succeed.
            </p>
            <Button variant="outline" className="w-full">
              View Documentation
            </Button>
          </div>
        </div>

        {/* Final CTA */}
        <div className="mt-20 text-center bg-gradient-to-r from-primary/5 to-secondary/5 rounded-2xl p-10 shadow-sm">
          <h3 className="text-2xl font-bold mb-4">Still have questions?</h3>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Can't find the answer you're looking for? Our friendly support team is here to help you optimize your AI costs and get the most out of Tokemizer.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Link href="/contact">
              <Button size="lg">Contact Support</Button>
            </Link>
            <Link href="/login?action=register">
              <Button variant="outline" size="lg">Get Started Free</Button>
            </Link>
          </div>
          <p className="text-sm text-muted-foreground mt-4">
            Average response time: 2 hours • 24/7 support available
          </p>
        </div>
      </div>
    </PublicLayout>
  );
}
