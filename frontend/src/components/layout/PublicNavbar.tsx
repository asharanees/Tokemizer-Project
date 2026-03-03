import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Zap } from "lucide-react";

export function PublicNavbar() {
    const [location] = useLocation();

    return (
        <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container flex h-16 items-center justify-between">
                <div className="flex items-center gap-2">
                    <Link href="/login" className="flex items-center gap-2">
                        <Zap className="h-6 w-6 text-primary fill-current" />
                        <span className="font-bold text-xl tracking-tight">Tokemizer</span>
                    </Link>
                </div>
                
                <nav className="hidden md:flex items-center gap-6 text-sm font-medium">
                    <Link href="/about" className={`transition-colors hover:text-foreground/80 ${location === "/about" ? "text-foreground" : "text-foreground/60"}`}>
                        About
                    </Link>
                    <Link href="/plans" className={`transition-colors hover:text-foreground/80 ${location === "/plans" ? "text-foreground" : "text-foreground/60"}`}>
                        Plans
                    </Link>
                    <Link href="/faq" className={`transition-colors hover:text-foreground/80 ${location === "/faq" ? "text-foreground" : "text-foreground/60"}`}>
                        FAQ
                    </Link>
                    <Link href="/contact" className={`transition-colors hover:text-foreground/80 ${location === "/contact" ? "text-foreground" : "text-foreground/60"}`}>
                        Contact
                    </Link>
                </nav>

                <div className="flex items-center gap-2">
                    {location !== "/login" && (
                         <Link href="/login">
                            <Button variant="ghost" size="sm">
                                Login
                            </Button>
                        </Link>
                    )}
                    {location !== "/login" && (
                        <Link href="/login?action=register">
                            <Button size="sm">Get Started</Button>
                        </Link>
                    )}
                </div>
            </div>
        </header>
    );
}
