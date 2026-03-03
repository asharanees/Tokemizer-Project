import { PublicLayout } from "@/components/layout/PublicLayout";
import { useSeo } from "@/lib/seo";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Mail, Phone, MessageCircle, Clock, Users, CheckCircle, Linkedin } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useState } from "react";

export default function Contact() {
  useSeo({
    title: "Contact Tokemizer Sales & Support",
    description: "Connect with Tokemizer for enterprise licensing, custom integration support, and volume pricing. Optimize your LLM infrastructure today.",
    path: "/contact",
  });

  const { toast } = useToast();
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    // Simulate network request
    await new Promise(resolve => setTimeout(resolve, 1500));
    setLoading(false);
    toast({
      title: "Message sent successfully!",
      description: "We'll get back to you within 2 hours.",
    });
    (e.target as HTMLFormElement).reset();
  };

  return (
    <PublicLayout>
      <div className="container mx-auto px-4 py-20 max-w-7xl">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-2 rounded-full text-sm font-medium mb-6">
            <MessageCircle className="h-4 w-4" />
            Get In Touch
          </div>
          <h1 className="text-4xl font-bold mb-4">Ready to Optimize Your AI Costs?</h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Whether you have questions about our technology, need enterprise pricing, or want to discuss a custom deployment,
            our team is here to help you succeed with Tokemizer.
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-12">
          {/* Contact Form */}
          <div className="lg:col-span-2">
            <Card className="shadow-xl border-0 bg-gradient-to-br from-card to-card/80">
              <CardHeader>
                <CardTitle className="text-2xl">Send us a message</CardTitle>
                <CardDescription className="text-base">
                  Tell us about your project and we'll get back to you with personalized recommendations.
                  Average response time: 2 hours.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="name">Full Name *</Label>
                      <Input
                        id="name"
                        placeholder="John Smith"
                        required
                        data-testid="input-contact-name"
                        className="h-12"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="email">Email Address *</Label>
                      <Input
                        id="email"
                        type="email"
                        placeholder="john@company.com"
                        required
                        data-testid="input-contact-email"
                        className="h-12"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="company">Company</Label>
                      <Input
                        id="company"
                        placeholder="Acme Corp"
                        data-testid="input-contact-company"
                        className="h-12"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="phone">Phone Number</Label>
                      <Input
                        id="phone"
                        type="tel"
                        placeholder="+1 (555) 123-4567"
                        data-testid="input-contact-phone"
                        className="h-12"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="subject">Subject *</Label>
                    <Input
                      id="subject"
                      placeholder="How can we help you?"
                      required
                      data-testid="input-contact-subject"
                      className="h-12"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="message">Message *</Label>
                    <Textarea
                      id="message"
                      placeholder="Tell us about your AI optimization needs, current challenges, or any questions you have about Tokemizer..."
                      className="min-h-[150px] resize-none"
                      required
                      data-testid="input-contact-message"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>How did you hear about us?</Label>
                    <select className="w-full h-12 px-3 border border-input rounded-md bg-background text-sm">
                      <option value="">Select an option</option>
                      <option value="search">Search Engine</option>
                      <option value="social">Social Media</option>
                      <option value="referral">Referral</option>
                      <option value="article">Article/Blog</option>
                      <option value="conference">Conference/Event</option>
                      <option value="other">Other</option>
                    </select>
                  </div>

                  <Button
                    type="submit"
                    className="w-full h-12 text-base font-semibold"
                    disabled={loading}
                    data-testid="button-contact-submit"
                  >
                    {loading ? "Sending..." : "Send Message"}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>

          {/* Contact Info & Support Options */}
          <div className="space-y-6">
            {/* Contact Methods */}
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Phone className="h-5 w-5 text-primary" />
                  Contact Methods
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-start gap-4">
                  <div className="bg-primary/10 p-3 rounded-lg flex-shrink-0">
                    <Mail className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-lg">Email Support</h3>
                    <p className="text-muted-foreground mb-1">support@tokemizer.com</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="bg-primary/10 p-3 rounded-lg flex-shrink-0">
                    <Linkedin className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-lg">LinkedIn</h3>
                    <a href="https://www.linkedin.com/company/111743477" target="_blank" rel="noreferrer" className="text-muted-foreground hover:text-primary transition-colors">
                      linkedin.com/company/111743477
                    </a>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="bg-primary/10 p-3 rounded-lg flex-shrink-0">
                    <MessageCircle className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-lg">Live Chat</h3>
                    <p className="text-muted-foreground mb-1">Coming Soon</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Support Options */}
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="h-5 w-5 text-primary" />
                  Support Options
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <span className="text-sm">Free tier support</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <span className="text-sm">Priority support (Pro+)</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <span className="text-sm">Dedicated success manager (Enterprise)</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <span className="text-sm">Custom integrations</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quick Stats */}
            <Card className="shadow-lg bg-gradient-to-br from-primary/5 to-secondary/5">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="h-5 w-5 text-primary" />
                  Why Choose Us?
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-center">
                  <div>
                    <div className="text-2xl font-bold text-primary">2hrs</div>
                    <div className="text-xs text-muted-foreground">Avg Response</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-primary">99.9%</div>
                    <div className="text-xs text-muted-foreground">Uptime</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-primary">70%</div>
                    <div className="text-xs text-muted-foreground">Upto 70% Token Savings</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-primary">24/7</div>
                    <div className="text-xs text-muted-foreground">Support</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Additional Help Section */}
        <div className="mt-16 grid md:grid-cols-3 gap-8">
          <Card className="text-center hover:shadow-md transition-shadow">
            <CardHeader>
              <div className="mx-auto w-12 h-12 bg-blue-500/10 rounded-lg flex items-center justify-center mb-4">
                <MessageCircle className="h-6 w-6 text-blue-500" />
              </div>
              <CardTitle className="text-lg">Documentation</CardTitle>
              <CardDescription>
                Need help with implementation details? Connect with our sales team for guidance.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="outline" className="w-full" asChild>
                <a href="mailto:sales@tokemizer.com">Contact Sales</a>
              </Button>
            </CardContent>
          </Card>

          <Card className="text-center hover:shadow-md transition-shadow">
            <CardHeader>
              <div className="mx-auto w-12 h-12 bg-green-500/10 rounded-lg flex items-center justify-center mb-4">
                <Linkedin className="h-6 w-6 text-green-500" />
              </div>
              <CardTitle className="text-lg">LinkedIn</CardTitle>
              <CardDescription>
                Follow our company profile for product updates, releases, and AI optimization insights.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="outline" className="w-full" asChild>
                <a href="https://www.linkedin.com/company/111743477" target="_blank" rel="noreferrer">
                  Follow on LinkedIn
                </a>
              </Button>
            </CardContent>
          </Card>

          <Card className="text-center hover:shadow-md transition-shadow">
            <CardHeader>
              <div className="mx-auto w-12 h-12 bg-purple-500/10 rounded-lg flex items-center justify-center mb-4">
                <Phone className="h-6 w-6 text-purple-500" />
              </div>
              <CardTitle className="text-lg">Enterprise Sales</CardTitle>
              <CardDescription>
                Custom deployments, volume pricing, and dedicated support
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="outline" className="w-full">
                Contact Sales
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </PublicLayout>
  );
}
