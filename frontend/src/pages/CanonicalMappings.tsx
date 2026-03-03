import { useMemo, useState } from "react";
import { useLocation } from "wouter";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { ToastAction } from "@/components/ui/toast";
import { authFetch } from "@/lib/authFetch";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Edit2, Trash2 } from "lucide-react";

interface Mapping {
  id: number;
  source_token: string;
  target_token: string;
  created_at: string;
  updated_at: string;
}

export default function CanonicalMappings() {
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [search, setSearch] = useState("");
  const [editingMapping, setEditingMapping] = useState<Mapping | null>(null);
  const [editSource, setEditSource] = useState("");
  const [editTarget, setEditTarget] = useState("");
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [, setLocation] = useLocation();

  const customMappingsQuery = useQuery({
    queryKey: ["custom-mappings"],
    queryFn: async (): Promise<Mapping[]> => {
      const res = await authFetch("/api/v1/mappings");
      if (!res.ok) throw new Error("Failed to load custom mappings");
      return res.json();
    },
  });

  const ootbMappingsQuery = useQuery({
    queryKey: ["ootb-mappings"],
    queryFn: async (): Promise<Mapping[]> => {
      const res = await authFetch("/api/v1/canonical-mappings?limit=500");
      if (!res.ok) throw new Error("Failed to load OOTB mappings");
      const data = await res.json();
      return data.mappings ?? [];
    },
  });

  const disabledOotbQuery = useQuery({
    queryKey: ["disabled-ootb"],
    queryFn: async (): Promise<string[]> => {
      const res = await authFetch("/api/v1/mappings/disabled-ootb");
      if (!res.ok) throw new Error("Failed to load preferences");
      const data = await res.json();
      return data.tokens ?? [];
    },
  });

  const upsertMutation = useMutation({
    mutationFn: async () => {
      const res = await authFetch("/api/v1/mappings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_token: source.trim(), target_token: target.trim() }),
      });
      if (!res.ok) throw new Error("Failed to save mapping");
      return res.json();
    },
    onSuccess: () => {
      setSource("");
      setTarget("");
      queryClient.invalidateQueries({ queryKey: ["custom-mappings"] });
      toast({
        title: "Mapping saved",
        description: "The new rule is now active in your optimizations.",
        action: (
          <ToastAction altText="Test in Playground" onClick={() => setLocation("/playground")}>
            Test Now
          </ToastAction>
        )
      });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (id: number) => {
      const res = await authFetch(`/api/v1/mappings/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Failed to delete mapping");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["custom-mappings"] });
      toast({ title: "Mapping deleted" });
    },
  });

  const editMutation = useMutation({
    mutationFn: async ({ id, source_token, target_token }: { id: number; source_token: string; target_token: string }) => {
      const res = await authFetch(`/api/v1/mappings/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_token: source_token.trim(), target_token: target_token.trim() }),
      });
      if (!res.ok) throw new Error("Failed to update mapping");
      return res.json();
    },
    onSuccess: () => {
      setEditingMapping(null);
      setEditSource("");
      setEditTarget("");
      queryClient.invalidateQueries({ queryKey: ["custom-mappings"] });
      toast({ title: "Mapping updated" });
    },
  });

  const handleEdit = (mapping: Mapping) => {
    setEditingMapping(mapping);
    setEditSource(mapping.source_token);
    setEditTarget(mapping.target_token);
  };

  const handleSaveEdit = () => {
    if (editingMapping && editSource.trim() && editTarget.trim()) {
      editMutation.mutate({
        id: editingMapping.id,
        source_token: editSource.trim(),
        target_token: editTarget.trim(),
      });
    }
  };

  const toggleOotbMutation = useMutation({
    mutationFn: async ({ source, enabled }: { source: string; enabled: boolean }) => {
      const res = await authFetch(`/api/v1/mappings/toggle-ootb?source_token=${encodeURIComponent(source)}&enabled=${enabled}`, {
        method: "POST"
      });
      if (!res.ok) throw new Error("Failed to toggle mapping");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["ootb-mappings"] });
      queryClient.invalidateQueries({ queryKey: ["disabled-ootb"] });
      toast({ title: "Preference updated" });
    }
  });

  const filteredCustom = useMemo(() => {
    const term = search.toLowerCase();
    return (customMappingsQuery.data ?? []).filter(m =>
      m.source_token.toLowerCase().includes(term) || m.target_token.toLowerCase().includes(term)
    );
  }, [customMappingsQuery.data, search]);

  const filteredOotb = useMemo(() => {
    const term = search.toLowerCase();
    return (ootbMappingsQuery.data ?? []).filter(m =>
      m.source_token.toLowerCase().includes(term) || m.target_token.toLowerCase().includes(term)
    );
  }, [ootbMappingsQuery.data, search]);

  const disabledOotbSet = useMemo(() => {
    return new Set((disabledOotbQuery.data ?? []).map(token => token.toLowerCase()));
  }, [disabledOotbQuery.data]);

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-glow">Canonical Mappings</h1>
          <p className="text-muted-foreground mt-1">
            Rules for normalizing tokens during optimization.
          </p>
        </div>

        <Tabs defaultValue="custom">
          <TabsList>
            <TabsTrigger value="custom">My Custom Mappings</TabsTrigger>
            <TabsTrigger value="ootb">Default Rules (OOTB)</TabsTrigger>
          </TabsList>

          <TabsContent value="custom" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Add Rule</CardTitle>
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
                  <Button onClick={() => upsertMutation.mutate()} disabled={upsertMutation.isPending} tooltip={upsertMutation.isPending ? "Saving rule..." : "Save this canonical mapping rule"}>
                    Save Rule
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Your Rules</CardTitle>
                <Input placeholder="Filter..." className="w-48 h-8" value={search} onChange={e => setSearch(e.target.value)} />
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
                    {filteredCustom.map(m => (
                      <TableRow key={m.id}>
                        <TableCell>{m.source_token}</TableCell>
                        <TableCell>{m.target_token}</TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-1">
                            <Button 
                              variant="ghost" 
                              size="icon" 
                              onClick={() => handleEdit(m)} 
                              tooltip="Edit this canonical mapping rule"
                            >
                              <Edit2 className="h-4 w-4" />
                            </Button>
                            <Button 
                              variant="ghost" 
                              size="icon" 
                              className="text-destructive" 
                              onClick={() => deleteMutation.mutate(m.id)} 
                              tooltip="Delete this canonical mapping rule"
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
          </TabsContent>

          <TabsContent value="ootb">
            <Card className="glass-card">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle>Standard Normalization Rules</CardTitle>
                  <CardDescription>These are applied globally. You can disable individual rules if needed.</CardDescription>
                </div>
                <Input placeholder="Filter..." className="w-48 h-8" value={search} onChange={e => setSearch(e.target.value)} />
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Source</TableHead>
                      <TableHead>Target</TableHead>
                      <TableHead className="text-right">Enable/Disable</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredOotb.map(m => (
                      <TableRow key={m.id}>
                        <TableCell>{m.source_token}</TableCell>
                        <TableCell>{m.target_token}</TableCell>
                        <TableCell className="text-right">
                          <Switch
                            checked={!disabledOotbSet.has(m.source_token.toLowerCase())}
                            onCheckedChange={(checked) => toggleOotbMutation.mutate({ source: m.source_token, enabled: checked })}
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Edit Dialog */}
      <Dialog open={!!editingMapping} onOpenChange={(open) => !open && setEditingMapping(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Mapping</DialogTitle>
            <DialogDescription>
              Update the source and target tokens for this canonical mapping rule.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Source</Label>
              <Input 
                value={editSource} 
                onChange={(e) => setEditSource(e.target.value)} 
                placeholder="Enter source token"
              />
            </div>
            <div className="space-y-2">
              <Label>Target</Label>
              <Input 
                value={editTarget} 
                onChange={(e) => setEditTarget(e.target.value)} 
                placeholder="Enter target token"
              />
            </div>
          </div>
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => setEditingMapping(null)}
              disabled={editMutation.isPending}
            >
              Cancel
            </Button>
            <Button 
              onClick={handleSaveEdit} 
              disabled={editMutation.isPending || !editSource.trim() || !editTarget.trim()}
            >
              {editMutation.isPending ? "Saving..." : "Save Changes"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Layout>
  );
}
