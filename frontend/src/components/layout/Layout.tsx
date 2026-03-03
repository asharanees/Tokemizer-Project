import { Sidebar } from "./Sidebar";
import { Bell, LogOut, Menu } from "lucide-react";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { useAuth } from "@/contexts/AuthContext";
import { CommandPalette } from "./CommandPalette";
import { Onboarding } from "./Onboarding";
import { useState } from "react";
import { Link } from "wouter";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";

export function Layout({ children }: { children: React.ReactNode }) {
  const [location] = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const { user, logout } = useAuth();
  const [, setLocation] = useLocation();

  const sectionLabel = (() => {
    if (location === "/") return "Dashboard";
    if (location.startsWith("/playground")) return "Optimizer";
    if (location.startsWith("/batch")) return "Batch Jobs";
    if (location.startsWith("/analytics")) return "Analytics";
    if (location.startsWith("/history")) return "History";
    if (location.startsWith("/canonical")) return "Canonical Maps";
    if (location.startsWith("/settings")) return "Settings";
    if (location.startsWith("/admin")) return "Admin Panel";
    return "Tokemizer";
  })();

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Onboarding />
      <CommandPalette />

      {/* Desktop Sidebar - Hidden on mobile */}
      <div className="hidden lg:block">
        <Sidebar />
      </div>

      {/* Mobile Sidebar - Shown on mobile */}
      <div className="lg:hidden">
        <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
          <SheetContent side="left" className="w-64 p-0 border-r border-sidebar-border">
            <Sidebar onNavigate={() => setMobileMenuOpen(false)} />
          </SheetContent>
        </Sheet>
      </div>

      <div className="lg:pl-64 flex flex-col min-h-screen">
        <header className="h-16 border-b border-border bg-background/80 backdrop-blur-md sticky top-0 z-40 px-4 lg:px-8 flex items-center justify-between">
          <div className="flex items-center gap-4 flex-1">
            {/* Mobile Menu Button */}
            <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="lg:hidden text-muted-foreground hover:text-primary" tooltip="Open mobile menu">
                  <Menu className="w-5 h-5" />
                </Button>
              </SheetTrigger>
            </Sheet>
            <span className="text-sm font-medium text-muted-foreground">{sectionLabel}</span>
          </div>

          <div className="flex items-center gap-2 lg:gap-4">
            <Button variant="ghost" size="icon" className="relative text-muted-foreground hover:text-primary" tooltip="View notifications">
              <Bell className="w-5 h-5" />
              <span className="absolute top-2 right-2 w-2 h-2 bg-primary rounded-full ring-2 ring-background"></span>
            </Button>
            <div className="h-8 w-[1px] bg-border hidden lg:block"></div>
            <div className="flex items-center gap-2 lg:gap-3">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button className="flex items-center gap-2 lg:gap-3 cursor-pointer hover:opacity-80 transition-opacity">
                    <div className="text-right hidden lg:block">
                      <p className="text-sm font-medium">{user?.name || "User"}</p>
                      <p className="text-xs text-muted-foreground capitalize">{user?.subscription_tier} Account</p>
                    </div>
                    <Avatar className="h-8 lg:h-9 w-8 lg:w-9 border border-primary/20">
                      <AvatarFallback className="bg-primary/10 text-primary font-bold text-xs lg:text-sm">
                        {user?.name?.substring(0, 2).toUpperCase() || "US"}
                      </AvatarFallback>
                    </Avatar>
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-56">
                  <div className="px-2 py-1.5">
                    <p className="text-sm font-medium">{user?.name}</p>
                    <p className="text-xs text-muted-foreground">{user?.email}</p>
                  </div>
                  <DropdownMenuSeparator />
                  <Link href="/profile">
                    <DropdownMenuItem className="cursor-pointer">
                      View Profile
                    </DropdownMenuItem>
                  </Link>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    className="cursor-pointer text-destructive focus:text-destructive focus:bg-destructive/10"
                    onClick={() => {
                      logout();
                      setLocation("/login");
                    }}
                  >
                    <LogOut className="w-4 h-4 mr-2" />
                    Logout
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
              <Button
                variant="ghost"
                size="icon"
                className="text-muted-foreground hover:text-destructive"
                onClick={() => {
                  logout();
                  setLocation("/login");
                }}
                tooltip="Logout"
              >
                <LogOut className="w-5 h-5" />
              </Button>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 animate-in fade-in duration-500">
          <div className="p-4 lg:p-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
