import { useEffect, useState } from "react";
import { useLocation } from "wouter";
import {
    CommandDialog,
    CommandEmpty,
    CommandGroup,
    CommandInput,
    CommandItem,
    CommandList,
    CommandSeparator,
} from "@/components/ui/command";
import {
    LayoutDashboard,
    Zap,
    History,
    Settings,
    User,
    Key,
    CreditCard,
    Map as MapIcon,
    Shield,
    FileCode
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

export function CommandPalette() {
    const [open, setOpen] = useState(false);
    const [, setLocation] = useLocation();
    const { user } = useAuth();

    useEffect(() => {
        const down = (e: KeyboardEvent) => {
            if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                setOpen((open) => !open);
            }
        };

        document.addEventListener("keydown", down);
        return () => document.removeEventListener("keydown", down);
    }, []);

    const runCommand = (command: () => void) => {
        setOpen(false);
        command();
    };

    return (
        <CommandDialog open={open} onOpenChange={setOpen}>
            <CommandInput placeholder="Type a command or search..." />
            <CommandList>
                <CommandEmpty>No results found.</CommandEmpty>
                <CommandGroup heading="Navigation">
                    <CommandItem onSelect={() => runCommand(() => setLocation("/"))}>
                        <LayoutDashboard className="mr-2 h-4 w-4" />
                        <span>Dashboard</span>
                    </CommandItem>
                    <CommandItem onSelect={() => runCommand(() => setLocation("/playground"))}>
                        <Zap className="mr-2 h-4 w-4" />
                        <span>Optimizer Playground</span>
                    </CommandItem>
                    <CommandItem onSelect={() => runCommand(() => setLocation("/history"))}>
                        <History className="mr-2 h-4 w-4" />
                        <span>History</span>
                    </CommandItem>
                    <CommandItem onSelect={() => runCommand(() => setLocation("/canonical"))}>
                        <MapIcon className="mr-2 h-4 w-4" />
                        <span>Canonical Maps</span>
                    </CommandItem>
                    <CommandItem onSelect={() => runCommand(() => setLocation("/keys"))}>
                        <Key className="mr-2 h-4 w-4" />
                        <span>API Keys</span>
                    </CommandItem>
                </CommandGroup>

                <CommandGroup heading="Account">
                    <CommandItem onSelect={() => runCommand(() => setLocation("/profile"))}>
                        <User className="mr-2 h-4 w-4" />
                        <span>Profile Settings</span>
                    </CommandItem>
                    <CommandItem onSelect={() => runCommand(() => setLocation("/subscription"))}>
                        <CreditCard className="mr-2 h-4 w-4" />
                        <span>Subscription & Billing</span>
                    </CommandItem>
                    <CommandItem onSelect={() => runCommand(() => setLocation("/settings"))}>
                        <Settings className="mr-2 h-4 w-4" />
                        <span>App Settings</span>
                    </CommandItem>
                </CommandGroup>

                {user?.role === "admin" && (
                    <>
                        <CommandSeparator />
                        <CommandGroup heading="Administration">
                            <CommandItem onSelect={() => runCommand(() => setLocation("/admin/users"))}>
                                <Shield className="mr-2 h-4 w-4 text-primary" />
                                <span>Manage Users</span>
                            </CommandItem>
                            <CommandItem onSelect={() => runCommand(() => setLocation("/admin/plans"))}>
                                <FileCode className="mr-2 h-4 w-4 text-primary" />
                                <span>Subscription Plans</span>
                            </CommandItem>
                            <CommandItem onSelect={() => runCommand(() => setLocation("/admin/ootb"))}>
                                <MapIcon className="mr-2 h-4 w-4 text-primary" />
                                <span>Global Mappings</span>
                            </CommandItem>
                            <CommandItem onSelect={() => runCommand(() => setLocation("/admin/settings"))}>
                                <Settings className="mr-2 h-4 w-4 text-primary" />
                                <span>System Configuration</span>
                            </CommandItem>
                            <CommandItem onSelect={() => runCommand(() => setLocation("/admin/tenant-health"))}>
                                <Shield className="mr-2 h-4 w-4 text-primary" />
                                <span>Tenant Health</span>
                            </CommandItem>
                        </CommandGroup>
                    </>
                )}
            </CommandList>
        </CommandDialog>
    );
}
