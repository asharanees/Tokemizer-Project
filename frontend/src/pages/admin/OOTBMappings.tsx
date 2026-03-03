import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";
import { authFetch } from "@/lib/authFetch";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Edit2, Trash2 } from "lucide-react";

interface GlobalMapping {
    id: number;
    source_token: string;
    target_token: string;
    created_at: string;
    updated_at: string;
}

export default function OOTBMappings() {
    const [source, setSource] = useState("");
    const [target, setTarget] = useState("");
    const [search, setSearch] = useState("");
    const [editingMapping, setEditingMapping] = useState<GlobalMapping | null>(null);
    const [editSource, setEditSource] = useState("");
    const [editTarget, setEditTarget] = useState("");
    const { toast } = useToast();
    const queryClient = useQueryClient();

    const mappingsQuery = useQuery({
        queryKey: ["admin-global-mappings"],
        queryFn: async (): Promise<GlobalMapping[]> => {
            const res = await authFetch("/api/v1/canonical-mappings?limit=5000");
            if (!res.ok) throw new Error("Failed to load global mappings");
            const data = await res.json();
            return data.mappings ?? [];
        },
    });

    const upsertMutation = useMutation({
        mutationFn: async () => {
            const res = await authFetch("/api/v1/canonical-mappings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ source_token: source.trim(), target_token: target.trim() }),
            });
            if (!res.ok) throw new Error("Failed to save global mapping");
            return res.json();
        },
        onSuccess: () => {
            setSource("");
            setTarget("");
            queryClient.invalidateQueries({ queryKey: ["admin-global-mappings"] });
            toast({ title: "Global mapping saved" });
        },
    });

    const deleteMutation = useMutation({
        mutationFn: async (id: number) => {
            const res = await authFetch("/api/v1/canonical-mappings", {
                method: "DELETE",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ids: [id] }),
            });
            if (!res.ok) throw new Error("Failed to delete mapping");
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["admin-global-mappings"] });
            toast({ title: "Global mapping deleted" });
        },
    });

    const handleEdit = (mapping: GlobalMapping) => {
        setEditingMapping(mapping);
        setEditSource(mapping.source_token);
        setEditTarget(mapping.target_token);
    };

    const handleSaveEdit = async () => {
        if (!editingMapping) return;
        
        try {
            const res = await authFetch("/api/v1/canonical-mappings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    source_token: editSource.trim(), 
                    target_token: editTarget.trim() 
                }),
            });
            
            if (!res.ok) throw new Error("Failed to save global mapping");
            
            await res.json();
            queryClient.invalidateQueries({ queryKey: ["admin-global-mappings"] });
            toast({ title: "Global mapping saved" });
            
            setEditingMapping(null);
            setEditSource("");
            setEditTarget("");
        } catch (error) {
            toast({ 
                title: "Error", 
                description: error instanceof Error ? error.message : "Failed to save mapping",
                variant: "destructive" 
            });
        }
    };

    const filteredMappings = useMemo(() => {
        const mappings = mappingsQuery.data ?? [];
        const term = search.toLowerCase();
        return mappings.filter(m =>
            m.source_token.toLowerCase().includes(term) ||
            m.target_token.toLowerCase().includes(term)
        );
    }, [mappingsQuery.data, search]);

    return (
        <Layout>
            <div className="space-y-6">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight text-glow">Default Mappings (OOTB)</h1>
                    <p className="text-muted-foreground mt-1">
                        Manage system-wide default canonicalization rules.
                    </p>
                </div>

                <Card className="glass-card">
                    <CardHeader>
                        <CardTitle>Add Global Rule</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <Label>Source</Label>
                                <Input value={source} onChange={e => setSource(e.target.value)} />
                            </div>
                            <div className="space-y-2">
                                <Label>Target</Label>
                                <Input value={target} onChange={e => setTarget(e.target.value)} />
                            </div>
                        </div>
                        <div className="flex justify-end">
                            <Button onClick={() => upsertMutation.mutate()} disabled={upsertMutation.isPending} tooltip={upsertMutation.isPending ? "Saving rule..." : "Save this global OOTB mapping rule"}>
                                {upsertMutation.isPending ? "Saving..." : "Save Global Rule"}
                            </Button>
                        </div>
                    </CardContent>
                </Card>

                <Card className="glass-card">
                    <CardHeader className="flex flex-row items-center justify-between">
                        <CardTitle>Global Rule Registry</CardTitle>
                        <Input
                            placeholder="Filter rules..."
                            className="w-64 h-8"
                            value={search}
                            onChange={e => setSearch(e.target.value)}
                        />
                    </CardHeader>
                    <CardContent>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Source</TableHead>
                                    <TableHead>Target</TableHead>
                                    <TableHead className="text-right">Actions</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {filteredMappings.map(m => (
                                    <TableRow key={m.id}>
                                        <TableCell className="font-medium">{m.source_token}</TableCell>
                                        <TableCell className="text-muted-foreground">{m.target_token}</TableCell>
                                        <TableCell className="text-right">
                                            <div className="flex items-center justify-end gap-1">
                                                <Button 
                                                    variant="ghost" 
                                                    size="icon" 
                                                    onClick={() => handleEdit(m)} 
                                                    tooltip="Edit this global OOTB mapping rule"
                                                >
                                                    <Edit2 className="h-4 w-4" />
                                                </Button>
                                                <Button 
                                                    variant="ghost" 
                                                    size="icon" 
                                                    className="text-destructive" 
                                                    onClick={() => deleteMutation.mutate(m.id)} 
                                                    tooltip="Delete this global OOTB mapping rule"
                                                >
                                                    <Trash2 className="h-4 w-4" />
                                                </Button>
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </CardContent>
                </Card>
            </div>

            <Dialog open={!!editingMapping} onOpenChange={(open) => !open && setEditingMapping(null)}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Edit Global Mapping</DialogTitle>
                        <DialogDescription>
                            Update the source and target tokens for this global canonical mapping rule.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="grid gap-4 py-4">
                        <div className="grid grid-cols-4 items-center gap-4">
                            <Label htmlFor="edit-source" className="text-right">
                                Source
                            </Label>
                            <Input
                                id="edit-source"
                                value={editSource}
                                onChange={(e) => setEditSource(e.target.value)}
                                className="col-span-3"
                            />
                        </div>
                        <div className="grid grid-cols-4 items-center gap-4">
                            <Label htmlFor="edit-target" className="text-right">
                                Target
                            </Label>
                            <Input
                                id="edit-target"
                                value={editTarget}
                                onChange={(e) => setEditTarget(e.target.value)}
                                className="col-span-3"
                            />
                        </div>
                    </div>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setEditingMapping(null)}>
                            Cancel
                        </Button>
                        <Button onClick={handleSaveEdit}>Save Changes</Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </Layout>
    );
}
