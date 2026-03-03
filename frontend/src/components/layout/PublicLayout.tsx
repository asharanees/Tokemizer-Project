import { PublicNavbar } from "./PublicNavbar";

interface PublicLayoutProps {
    children: React.ReactNode;
}

export function PublicLayout({ children }: PublicLayoutProps) {
    return (
        <div className="min-h-screen flex flex-col bg-background">
            <PublicNavbar />
            <main className="flex-1">
                {children}
            </main>
            <footer className="border-t py-6 md:py-0">
                <div className="container flex flex-col items-center justify-between gap-4 md:h-24 md:flex-row">
                    <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
                        © {new Date().getFullYear()} Tokemizer. All rights reserved.
                    </p>
                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                        <a href="#" className="hover:underline">Privacy Policy</a>
                        <a href="#" className="hover:underline">Terms of Service</a>
                    </div>
                </div>
            </footer>
        </div>
    );
}
