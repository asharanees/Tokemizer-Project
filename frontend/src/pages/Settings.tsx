import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { RefreshCw } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/hooks/use-toast";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { HelpTooltip } from "@/components/ui/HelpTooltip";
import { NavigationBreadcrumb } from "@/components/layout/NavigationBreadcrumb";
import { Separator } from "@/components/ui/separator";
import { authFetch } from "@/lib/authFetch";

interface LLMProfilePublic {
  name: string;
  provider: string;
  model: string;
  has_api_key: boolean;
}

interface LLMProfileForm {
  name: string;
  provider: string;
  model: string;
  api_key?: string;
  has_api_key?: boolean;
}

interface SettingsResponse {
  semantic_guard_threshold: number;
  semantic_guard_enabled: boolean;
  semantic_guard_model?: string | null;
  guard_latency_ms: number;
  guard_tokens_saved: number;
  telemetry_baseline_window_days: number;
  optimizer_cache_size: number;
  telemetry_enabled: boolean;
  lsh_enabled: boolean;
  lsh_similarity_threshold: number;
  llm_system_context: string;
  llm_profiles: LLMProfilePublic[];
}

type SettingsFormValues = Omit<SettingsResponse, "llm_profiles"> & {
  semantic_guard_model?: string;
  llm_profiles: LLMProfileForm[];
};

type BooleanFieldKey = "semantic_guard_enabled" | "telemetry_enabled" | "lsh_enabled";
type NumberFieldKey = "semantic_guard_threshold" | "lsh_similarity_threshold" | "optimizer_cache_size";
type GuardNumberFieldKey =
  | "guard_latency_ms"
  | "guard_tokens_saved"
  | "telemetry_baseline_window_days";

const booleanFields: { key: BooleanFieldKey; label: string; hint?: string }[] = [
  { key: "semantic_guard_enabled", label: "Semantic Guard Enabled", hint: "Protects meaning via similarity checks." },
  { key: "telemetry_enabled", label: "Telemetry Enabled", hint: "Collect per-pass timing metrics." },
  { key: "lsh_enabled", label: "LSH Deduplication Enabled", hint: "Uses MinHash to detect near-duplicates." },
];

const numberFields: { key: NumberFieldKey; label: string; hint?: string; step?: string; min?: number; max?: number }[] = [
  { key: "semantic_guard_threshold", label: "Semantic Guard Threshold", hint: "Similarity required before accepting reductions (0-1).", step: "0.001", min: 0, max: 1 },
  { key: "lsh_similarity_threshold", label: "LSH Similarity Threshold", hint: "Higher values make deduplication stricter (0-1).", step: "0.001", min: 0, max: 1 },
  { key: "optimizer_cache_size", label: "Optimization Cache Size", hint: "LRU size for fast-path reuse. Updating clears the cache.", step: "1", min: 0 },
];

const guardFields: { key: GuardNumberFieldKey; label: string; hint: string; step?: string; min?: number }[] = [
  {
    key: "guard_latency_ms",
    label: "Max latency guard (ms)",
    hint: "Ensure cumulative optimizer latency per request stays below this value.",
    step: "10",
    min: 0,
  },
  {
    key: "guard_tokens_saved",
    label: "Token savings baseline",
    hint: "Average tokens saved per run should remain above this threshold.",
    step: "1",
    min: 0,
  },
  {
    key: "telemetry_baseline_window_days",
    label: "Telemetry baseline window (days)",
    hint: "Duration used when sampling telemetry/history to evaluate guardrails.",
    step: "1",
    min: 1,
  },
];

export default function Settings() {
  const { data, isLoading, refetch, isFetching, isError } = useQuery<SettingsResponse>({
    queryKey: ["settings"],
    queryFn: async () => {
      const res = await authFetch("/api/v1/settings");
      if (!res.ok) throw new Error("Failed to fetch settings");
      return res.json();
    },
    refetchInterval: 30000,
  });
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [formValues, setFormValues] = useState<SettingsFormValues | null>(null);
  const [isDirty, setIsDirty] = useState(false);
  const [profileName, setProfileName] = useState("");
  const [profileProvider, setProfileProvider] = useState("");
  const [profileModel, setProfileModel] = useState("");
  const [profileApiKey, setProfileApiKey] = useState("");

  useEffect(() => {
    if (data && !isDirty) {
      setFormValues({
        ...data,
        semantic_guard_model: data.semantic_guard_model ?? "",
        llm_profiles: (data.llm_profiles ?? []).map((profile) => ({
          name: profile.name,
          provider: profile.provider,
          model: profile.model,
          has_api_key: profile.has_api_key,
        })),
      });
    }
  }, [data, isDirty]);

  const saveMutation = useMutation({
    mutationFn: async () => {
      if (!formValues) {
        throw new Error("Settings not loaded");
      }
      const payload = {
        ...formValues,
        semantic_guard_model: formValues.semantic_guard_model?.trim() || null,
        llm_profiles: (formValues.llm_profiles ?? []).map(({ name, provider, model, api_key }) => {
          const trimmed = (api_key ?? "").trim();
          return {
            name,
            provider,
            model,
            ...(trimmed ? { api_key: trimmed } : {}),
          };
        }),
      };
      const res = await authFetch("/api/v1/settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("Failed to save settings");
      return res.json() as Promise<SettingsResponse>;
    },
    onSuccess: (updated) => {
      setFormValues({
        ...updated,
        semantic_guard_model: updated.semantic_guard_model ?? "",
        llm_profiles: (updated.llm_profiles ?? []).map((profile) => ({
          name: profile.name,
          provider: profile.provider,
          model: profile.model,
          has_api_key: profile.has_api_key,
        })),
      });
      queryClient.invalidateQueries({ queryKey: ["settings"] });
      setIsDirty(false);
      toast({ title: "Settings saved", description: "Runtime configuration updated." });
    },
    onError: () => {
      toast({ title: "Save failed", description: "Unable to update settings. Try again.", variant: "destructive" });
    },
  });

  const handleRefresh = async () => {
    setIsDirty(false);
    const result = await refetch();
    if (result.data) {
      setFormValues({
        ...result.data,
        semantic_guard_model: result.data.semantic_guard_model ?? "",
        llm_profiles: (result.data.llm_profiles ?? []).map((profile) => ({
          name: profile.name,
          provider: profile.provider,
          model: profile.model,
          has_api_key: profile.has_api_key,
        })),
      });
    }
  };

  return (
    <Layout>
      <div className="space-y-6 max-w-4xl w-full">
        <NavigationBreadcrumb />
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight font-display text-glow">Settings</h1>
            <p className="text-xs sm:text-sm text-muted-foreground mt-1">Runtime configuration as currently applied on the backend.</p>
          </div>
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 w-full sm:w-auto">
            <Button variant="outline" size="sm" onClick={handleRefresh} disabled={isFetching} className="w-full sm:w-auto" tooltip={isFetching ? "Refreshing settings..." : "Refresh settings from server"}>
              {isFetching ? <RefreshCw className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
              <span className="ml-2 hidden sm:inline">Refresh</span>
            </Button>
            <Button size="sm" onClick={() => saveMutation.mutate()} disabled={isLoading || !formValues || saveMutation.isPending} className="w-full sm:w-auto" tooltip={saveMutation.isPending ? "Saving settings..." : "Save your settings changes"}>
              {saveMutation.isPending ? "Saving..." : "Save Changes"}
            </Button>
          </div>
        </div>

        <Tabs defaultValue="general" className="w-full">
          <TabsList className="mb-4 w-full sm:w-auto justify-start bg-muted/50 p-1">
            <TabsTrigger value="general" className="text-xs sm:text-sm">General Configuration</TabsTrigger>
            <TabsTrigger value="profiles" className="text-xs sm:text-sm">LLM Profiles</TabsTrigger>
          </TabsList>

          <TabsContent value="general" className="space-y-4 lg:space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg lg:text-base">
                  Optimizer Defaults
                  <HelpTooltip content="Global defaults for optimization runs." />
                </CardTitle>
                <CardDescription className="text-xs sm:text-sm">Values are derived from runtime configuration.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 lg:space-y-6">
                {isError ? (
                  <div className="rounded-lg border border-destructive/40 bg-white p-3 lg:p-4 text-xs sm:text-sm text-destructive">
                    Unable to load settings. Check the backend logs and refresh.
                  </div>
                ) : isLoading || !formValues ? (
                  <div className="grid md:grid-cols-2 gap-4">
                    {Array.from({ length: 6 }).map((_, index) => (
                      <Skeleton key={index} className="h-20" />
                    ))}
                  </div>
                ) : (
                  <>
                    <div className="grid md:grid-cols-2 gap-4">
                      <div className="space-y-2 md:col-span-2">
                        <Label htmlFor="llm_system_context" className="flex items-center gap-1.5">
                          LLM System Context
                          <HelpTooltip content="Global instruction context prepended to user prompts when LLM-based optimization is selected." />
                        </Label>
                        <Textarea
                          id="llm_system_context"
                          rows={8}
                          placeholder="Enter global LLM optimization instructions"
                          value={formValues.llm_system_context ?? ""}
                          onChange={(event) =>
                            setFormValues((prev) => {
                              if (!prev) return prev;
                              setIsDirty(true);
                              return { ...prev, llm_system_context: event.target.value };
                            })
                          }
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="semantic_guard_model" className="flex items-center gap-1.5">
                          Semantic Guard Model
                          <HelpTooltip content="The model used to calculate semantic similarity for guardrails." />
                        </Label>
                        <Input
                          id="semantic_guard_model"
                          placeholder="BAAI/bge-small-en-v1.5"
                          value={formValues.semantic_guard_model ?? ""}
                          onChange={(event) =>
                            setFormValues((prev) => {
                              if (!prev) return prev;
                              setIsDirty(true);
                              return { ...prev, semantic_guard_model: event.target.value };
                            })
                          }
                        />
                      </div>
                      {numberFields.map(({ key, label, hint, step, min, max }) => (
                        <div key={key} className="space-y-2">
                          <Label htmlFor={key} className="flex items-center gap-1.5">
                            {label}
                            {hint && <HelpTooltip content={hint} />}
                          </Label>
                          <Input
                            id={key}
                            type="number"
                            step={step}
                            min={min}
                            max={max}
                            value={formValues[key] ?? 0}
                            onChange={(event) => {
                              const raw = event.target.value;
                              const parsed = raw === "" ? 0 : Number(raw);
                              if (Number.isNaN(parsed)) return;
                              setFormValues((prev) => {
                                if (!prev) return prev;
                                setIsDirty(true);
                                return { ...prev, [key]: parsed };
                              });
                            }}
                          />
                        </div>
                      ))}
                    </div>

                    <Separator />

                    <div className="grid md:grid-cols-2 gap-4">
                      {booleanFields.map(({ key, label, hint }) => (
                        <div key={key} className="flex items-center justify-between rounded-lg border border-border/70 bg-muted/10 p-3">
                          <div>
                            <p className="text-sm font-medium flex items-center gap-1.5">
                              {label}
                              {hint && <HelpTooltip content={hint} />}
                            </p>
                          </div>
                          <Switch
                            checked={Boolean(formValues[key])}
                            onCheckedChange={(value) =>
                              setFormValues((prev) => {
                                if (!prev) return prev;
                                setIsDirty(true);
                                return { ...prev, [key]: value };
                              })
                            }
                          />
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg lg:text-base">
                  Guardrails
                  <HelpTooltip content="Thresholds that protect latency, token savings, and sampling windows." />
                </CardTitle>
                <CardDescription className="text-xs sm:text-sm">
                  Tweak the guard thresholds to match your deployment SLA.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {isError ? (
                  <div className="rounded-lg border border-destructive/40 bg-white p-3 lg:p-4 text-xs sm:text-sm text-destructive">
                    Unable to load guardrail settings. Check the backend logs and refresh.
                  </div>
                ) : isLoading || !formValues ? (
                  <div className="grid md:grid-cols-3 gap-4">
                    {Array.from({ length: guardFields.length }).map((_, index) => (
                      <Skeleton key={index} className="h-20" />
                    ))}
                  </div>
                ) : (
                  <div className="grid md:grid-cols-3 gap-4">
                    {guardFields.map(({ key, label, hint, step, min }) => (
                      <div key={key} className="space-y-2">
                        <Label htmlFor={key} className="flex items-center gap-1.5">
                          {label}
                          <HelpTooltip content={hint} />
                        </Label>
                        <Input
                          id={key}
                          type="number"
                          step={step}
                          min={min}
                          value={formValues[key] ?? 0}
                          onChange={(event) => {
                            const raw = event.target.value;
                            const parsed = raw === "" ? 0 : Number(raw);
                            if (Number.isNaN(parsed)) return;
                            setFormValues((prev) => {
                              if (!prev) return prev;
                              setIsDirty(true);
                              return { ...prev, [key]: parsed };
                            });
                          }}
                        />
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="profiles">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>LLM Profiles</CardTitle>
                <CardDescription>Store provider/model/API key combos for reuse in the playground.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="profile-name">Profile Name</Label>
                    <Input
                      id="profile-name"
                      placeholder="e.g. OpenAI Prod"
                      value={profileName}
                      onChange={(event) => setProfileName(event.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="profile-provider">Provider</Label>
                    <Input
                      id="profile-provider"
                      placeholder="openai"
                      value={profileProvider}
                      onChange={(event) => setProfileProvider(event.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="profile-model">Model</Label>
                    <Input
                      id="profile-model"
                      placeholder="gpt-5"
                      value={profileModel}
                      onChange={(event) => setProfileModel(event.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="profile-api-key">API Key</Label>
                    <Input
                      id="profile-api-key"
                      type="password"
                      placeholder="Paste API key"
                      value={profileApiKey}
                      onChange={(event) => setProfileApiKey(event.target.value)}
                    />
                  </div>
                </div>
                <Button
                  tooltip="Add a new LLM profile with API credentials"
                  onClick={() => {
                    const name = profileName.trim();
                    const provider = profileProvider.trim();
                    const model = profileModel.trim();
                    const apiKey = profileApiKey.trim();
                    if (!name || !provider || !model || !apiKey) {
                      toast({
                        title: "Missing profile details",
                        description: "Provide name, provider, model, and API key.",
                        variant: "destructive",
                      });
                      return;
                    }
                    setFormValues((prev) => {
                      if (!prev) return prev;
                      setIsDirty(true);
                      return {
                        ...prev,
                        llm_profiles: [
                          ...prev.llm_profiles.filter((profile) => profile.name !== name),
                          { name, provider, model, api_key: apiKey, has_api_key: true },
                        ],
                      };
                    });
                    setProfileName("");
                    setProfileProvider("");
                    setProfileModel("");
                    setProfileApiKey("");
                  }}
                >
                  Add Profile
                </Button>
                {formValues?.llm_profiles?.length ? (
                  <div className="space-y-3">
                    {formValues.llm_profiles.map((profile) => (
                      <div key={profile.name} className="flex items-center justify-between rounded-lg border border-border/70 bg-muted/10 p-3">
                        <div>
                          <p className="text-sm font-medium">{profile.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {profile.provider} · {profile.model}
                            {profile.has_api_key || Boolean(profile.api_key?.trim()) ? " · key stored" : ""}
                          </p>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-destructive hover:text-destructive"
                          tooltip="Remove this LLM profile"
                          onClick={() =>
                            setFormValues((prev) => {
                              if (!prev) return prev;
                              setIsDirty(true);
                              return {
                                ...prev,
                                llm_profiles: prev.llm_profiles.filter((item) => item.name !== profile.name),
                              };
                            })
                          }
                        >
                          Remove
                        </Button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No saved profiles yet.</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </Layout>
  );
}
