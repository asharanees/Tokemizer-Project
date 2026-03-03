import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { ToastAction } from "@/components/ui/toast";
import { Loader2, Trash2, Copy, Plus } from "lucide-react";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog";
import { authFetch } from "@/lib/authFetch";

interface ApiKey {
    id: string;
    name: string;
    prefix: string;
    created_at: string;
    last_used_at: string | null;
    is_active: boolean;
}

interface NewApiKeyResponse {
    id: string;
    name: string;
    key: string;
    created_at: string;
}

export default function ApiKeys() {
    const { data: keys, isLoading } = useQuery<ApiKey[]>({
        queryKey: ["api-keys"],
        queryFn: async () => {
            const res = await authFetch("/api/v1/keys");
            if (!res.ok) throw new Error("Failed to fetch keys");
            return res.json();
        },
    });

    const [newKeyName, setNewKeyName] = useState("");
    const [newKey, setNewKey] = useState<NewApiKeyResponse | null>(null);
    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const { toast } = useToast();
    const queryClient = useQueryClient();

    const createMutation = useMutation({
        mutationFn: async (name: string) => {
            const res = await authFetch("/api/v1/keys", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ name }),
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Failed to create key");
            }
            return res.json();
        },
        onSuccess: (data: NewApiKeyResponse) => {
            setNewKey(data);
            queryClient.invalidateQueries({ queryKey: ["api-keys"] });
            setNewKeyName("");
            setIsDialogOpen(false);
            toast({
                title: "API Key created",
                description: "Save it now! You won't see it again.",
                action: (
                    <ToastAction
                        altText="Copy key"
                        onClick={() => copyToClipboard(data.key)}
                    >
                        Copy
                    </ToastAction>
                )
            });
        },
        onError: (err: Error) => {
            toast({ title: "Error", description: err.message, variant: "destructive" });
        },
    });

    const deleteMutation = useMutation({
        mutationFn: async (id: string) => {
            const res = await authFetch(`/api/v1/keys/${id}`, {
                method: "DELETE",
            });
            if (!res.ok) throw new Error("Failed to delete key");
            return res.json();
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["api-keys"] });
            toast({ title: "API Key deleted", description: "The key has been revoked." });
        },
    });

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
        toast({ title: "Copied!", description: "API Key copied to clipboard." });
    };

    return (
        <Layout>
            <div className="space-y-6 max-w-4xl w-full">
                <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
                    <div>
                        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight font-display text-glow">API Keys</h1>
                        <p className="text-xs sm:text-sm text-muted-foreground mt-1">Manage your API keys for accessing Tokemizer programmatically.</p>
                    </div>
                    <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                        <DialogTrigger asChild>
                            <Button>
                                <Plus className="w-4 h-4 mr-2" /> Create New Key
                            </Button>
                        </DialogTrigger>
                        <DialogContent>
                            <DialogHeader>
                                <DialogTitle>Create API Key</DialogTitle>
                                <DialogDescription>
                                    Give your key a name to identify it later.
                                </DialogDescription>
                            </DialogHeader>
                            <div className="space-y-4 py-4">
                                <div className="space-y-2">
                                    <Label htmlFor="key-name">Key Name</Label>
                                    <Input
                                        id="key-name"
                                        placeholder="e.g. Production Server"
                                        value={newKeyName}
                                        onChange={(e) => setNewKeyName(e.target.value)}
                                    />
                                </div>
                            </div>
                            <DialogFooter>
                                <Button onClick={() => createMutation.mutate(newKeyName)} disabled={createMutation.isPending || !newKeyName} tooltip={createMutation.isPending ? "Creating API key..." : "Create new API key"}>
                                    {createMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : null}
                                    Create
                                </Button>
                            </DialogFooter>
                        </DialogContent>
                    </Dialog>
                </div>

                {newKey && (
                    <Card className="border-green-500/50 bg-green-500/10 dark:bg-green-900/20">
                        <CardHeader>
                            <CardTitle className="text-green-700 dark:text-green-400">New API Key Generated</CardTitle>
                            <CardDescription>
                                Please copy this key immediately. You won't be able to see it again!
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="flex items-center gap-2">
                                <code className="flex-1 bg-background p-3 rounded border font-mono text-sm break-all">
                                    {newKey.key}
                                </code>
                                <Button size="icon" variant="outline" onClick={() => copyToClipboard(newKey.key)} tooltip="Copy API key to clipboard">
                                    <Copy className="w-4 h-4" />
                                </Button>
                            </div>
                            <Button variant="ghost" size="sm" className="mt-4" onClick={() => setNewKey(null)} tooltip="Confirm you have saved the API key">
                                I have saved it
                            </Button>
                        </CardContent>
                    </Card>
                )}

                <Card className="glass-card">
                    <CardHeader>
                        <CardTitle>Your Keys</CardTitle>
                    </CardHeader>
                    <CardContent>
                        {isLoading ? (
                            <div className="space-y-4">
                                {[1, 2, 3].map(i => <div key={i} className="h-12 bg-muted/50 rounded animate-pulse" />)}
                            </div>
                        ) : keys?.length === 0 ? (
                            <p className="text-muted-foreground">No API keys found. Create one to get started.</p>
                        ) : (
                            <div className="space-y-4">
                                {keys?.map((key) => (
                                    <div key={key.id} className="flex items-center justify-between p-4 rounded-lg border bg-card/50">
                                        <div>
                                            <h3 className="font-medium">{key.name}</h3>
                                            <p className="text-xs text-muted-foreground font-mono mt-1">
                                                {key.prefix} • Created {new Date(key.created_at).toLocaleDateString()}
                                                {key.last_used_at && ` • Last used ${new Date(key.last_used_at).toLocaleDateString()}`}
                                            </p>
                                        </div>
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="text-destructive hover:text-destructive hover:bg-destructive/10"
                                            onClick={() => deleteMutation.mutate(key.id)}
                                            disabled={deleteMutation.isPending}
                                            tooltip="Delete this API key"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </Button>
                                    </div>
                                ))}
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </Layout>
    );
}
