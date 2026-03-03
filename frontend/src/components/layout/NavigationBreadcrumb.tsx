import { ChevronRight, Home } from "lucide-react";
import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";

export function NavigationBreadcrumb() {
    const [location] = useLocation();
    const pathSegments = location.split("/").filter(Boolean);

    return (
        <nav className="flex items-center space-x-1.5 text-xs text-muted-foreground mb-6">
            <Link href="/">
                <div className="flex items-center hover:text-foreground transition-colors cursor-pointer">
                    <Home className="h-3.5 w-3.5 mr-1" />
                    <span>Home</span>
                </div>
            </Link>

            {pathSegments.map((segment, index) => {
                const url = `/${pathSegments.slice(0, index + 1).join("/")}`;
                const isLast = index === pathSegments.length - 1;

                return (
                    <div key={url} className="flex items-center space-x-1.5">
                        <ChevronRight className="h-3 w-3 text-muted-foreground/30" />
                        <Link href={url}>
                            <span className={cn(
                                "capitalize transition-colors cursor-pointer",
                                isLast ? "text-foreground font-medium pointer-events-none" : "hover:text-foreground"
                            )}>
                                {segment.replace(/-/g, " ")}
                            </span>
                        </Link>
                    </div>
                );
            })}
        </nav>
    );
}
