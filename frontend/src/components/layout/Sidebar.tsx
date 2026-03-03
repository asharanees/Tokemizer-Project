import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  Zap,
  Settings,
  History,
  BarChart3,
  Terminal,
  Layers,
  Map as MapIcon,
  User,
  Key,
  CreditCard,
  Users,
  ShieldAlert,
  Sliders,
  Database,
  Activity,
  BookOpen
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { QuotaWidget } from "./QuotaWidget";

interface SidebarProps {
  onNavigate?: () => void;
}

export function Sidebar({ onNavigate }: SidebarProps) {
  const [location] = useLocation();
  const { user } = useAuth();

  const navItems = [
    { icon: LayoutDashboard, label: "Dashboard", href: "/" },
    { icon: Zap, label: "Optimizer", href: "/playground" },
    { icon: Layers, label: "Batch Jobs", href: "/batch" },
    { icon: BarChart3, label: "Analytics", href: "/analytics" },
    { icon: History, label: "History", href: "/history" },
    { icon: MapIcon, label: "Canonical Maps", href: "/canonical" },
    { icon: Key, label: "API Keys", href: "/keys" },
    { icon: CreditCard, label: "Subscription", href: "/subscription" },
    { icon: User, label: "Profile", href: "/profile" },
    { icon: Settings, label: "Settings", href: "/settings" },
  ];

  const adminItems = [
    { icon: Users, label: "Manage Users", href: "/admin/users" },
    { icon: ShieldAlert, label: "Subscription Plans", href: "/admin/plans" },
    { icon: MapIcon, label: "Global Mappings", href: "/admin/ootb" },
    { icon: Database, label: "Model Management", href: "/admin/models" },
    { icon: Activity, label: "Tenant Health", href: "/admin/tenant-health" },
    { icon: Sliders, label: "System Settings", href: "/admin/settings" },
    { icon: BookOpen, label: "API Documentation", href: "/api/v1/docs", external: true },
  ];

  return (
    <aside className="w-full lg:w-64 h-screen border-r border-sidebar-border bg-sidebar flex flex-col fixed lg:fixed left-0 top-0 z-50 lg:z-50">
      <div className="flex flex-col items-center justify-center gap-2 px-4 py-6 border-b border-sidebar-border">
        <span className="font-display font-bold text-2xl lg:text-3xl tracking-tight text-primary">
          TOKEMIZER
        </span>
        <img
          src="/fav.png"
          alt="Tokemizer"
          className="mt-2 w-3/4 max-w-[12rem] h-auto rounded-sm object-contain"
        />
      </div>

      <nav className="flex-1 py-4 lg:py-6 px-3 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const isActive = location === item.href;
          return (
            <Link key={item.href} href={item.href}>
              <div 
                onClick={() => onNavigate?.()}
                className={cn(
                "flex items-center gap-3 px-3 py-2 lg:py-2.5 rounded-md text-sm font-medium transition-all duration-200 group cursor-pointer",
                isActive
                  ? "bg-sidebar-accent text-primary border-l-2 border-primary"
                  : "text-sidebar-foreground/70 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
              )}>
                <item.icon className={cn(
                  "w-5 h-5 transition-colors shrink-0",
                  isActive ? "text-primary" : "text-sidebar-foreground/50 group-hover:text-sidebar-foreground"
                )} />
                <span className="truncate">{item.label}</span>
              </div>
            </Link>
          );
        })}

        {user?.role === "admin" && (
          <>
            <div className="pt-4 pb-2 px-3">
              <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/50">Admin Panel</span>
            </div>
            {adminItems.map((item) => {
              const isActive = !item.external && location === item.href;
              const content = (
                <div 
                  onClick={() => onNavigate?.()}
                  className={cn(
                  "flex items-center gap-3 px-3 py-2 lg:py-2.5 rounded-md text-sm font-medium transition-all duration-200 group cursor-pointer",
                  isActive
                    ? "bg-sidebar-accent text-primary border-l-2 border-primary"
                    : "text-sidebar-foreground/70 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
                )}>
                  <item.icon className={cn(
                    "w-5 h-5 transition-colors shrink-0",
                    isActive ? "text-primary" : "text-sidebar-foreground/50 group-hover:text-sidebar-foreground"
                  )} />
                  <span className="truncate">{item.label}</span>
                </div>
              );

              if (item.external) {
                return (
                  <a key={item.href} href={item.href} target="_blank" rel="noreferrer">
                    {content}
                  </a>
                );
              }

              return (
                <Link key={item.href} href={item.href}>
                  {content}
                </Link>
              );
            })}
          </>
        )}
      </nav>

      <div className="p-3 lg:p-4 border-t border-sidebar-border space-y-3 lg:space-y-4">
        {user && user.role !== "admin" && <QuotaWidget />}
        <div className="bg-sidebar-accent/50 rounded-lg p-3 lg:p-4 border border-sidebar-border">
          <div className="flex items-center gap-2 mb-2">
            <Terminal className="w-4 h-4 text-primary shrink-0" />
            <span className="text-xs font-mono text-muted-foreground">API Status</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse shrink-0"></div>
            <span className="text-xs font-medium text-green-400">Operational</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
