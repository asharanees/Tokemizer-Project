import { useEffect, useMemo, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { authFetch } from "@/lib/authFetch";

interface User {
    id: string;
    name: string;
    email: string;
    role: string;
    is_active: boolean;
    subscription_status: string;
    subscription_tier: string;
    created_at: string;
    quota_overage_bonus?: number;
}

interface UserListResponse {
    users: User[];
    total: number;
}

interface Plan {
    id: string;
    name: string;
    is_active: boolean;
}

interface CreateUserForm {
    name: string;
    email: string;
    password: string;
    role: "customer" | "admin";
    subscription_tier: string;
}

const CREATE_USER_FORM_INITIAL: CreateUserForm = {
    name: "",
    email: "",
    password: "",
    role: "customer",
    subscription_tier: "",
};

const parseResponseError = async (res: Response) => {
    try {
        const body = await res.json();
        if (body?.detail) {
            return body.detail;
        }
        if (body?.message) {
            return body.message;
        }
    } catch {
        // ignore parse errors
    }
    return res.statusText || "An unexpected error occurred";
};

export default function Users() {
    const { toast } = useToast();
    const queryClient = useQueryClient();
    const [selectedUser, setSelectedUser] = useState<User | null>(null);
    const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
    const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
    const [createForm, setCreateForm] = useState<CreateUserForm>(
        CREATE_USER_FORM_INITIAL
    );
    const [deletingUserId, setDeletingUserId] = useState<string | null>(null);

    const { data, isLoading } = useQuery<UserListResponse>({
        queryKey: ["admin-users"],
        queryFn: async () => {
            const res = await authFetch("/api/admin/users");
            if (!res.ok) throw new Error("Failed to fetch users");
            return res.json();
        },
    });

    const { data: plans } = useQuery<Plan[]>({
        queryKey: ["admin-plans"],
        queryFn: async () => {
            const res = await authFetch("/api/admin/plans");
            if (!res.ok) throw new Error("Failed to fetch plans");
            return res.json();
        },
    });

    const planNameById = useMemo(() => {
        if (!plans) {
            return {};
        }
        return plans.reduce<Record<string, string>>((acc, plan) => {
            acc[plan.id] = plan.name;
            return acc;
        }, {});
    }, [plans]);

    useEffect(() => {
        if (
            isCreateDialogOpen &&
            createForm.role === "customer" &&
            !createForm.subscription_tier &&
            plans?.length
        ) {
            setCreateForm((prev) => ({ ...prev, subscription_tier: plans[0].id }));
        }
    }, [plans, createForm.role, createForm.subscription_tier, isCreateDialogOpen]);

    const resetCreateForm = () => {
        setCreateForm({ ...CREATE_USER_FORM_INITIAL });
    };

    const openCreateDialog = () => {
        resetCreateForm();
        setIsCreateDialogOpen(true);
    };

    const closeCreateDialog = () => {
        setIsCreateDialogOpen(false);
        resetCreateForm();
    };

    const updateMutation = useMutation({
        mutationFn: async (updatedUser: Partial<User>) => {
            if (!selectedUser?.id) {
                throw new Error("No user selected");
            }
            const res = await authFetch(`/api/admin/users/${selectedUser?.id}`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(updatedUser),
            });
            if (!res.ok) {
                throw new Error(await parseResponseError(res));
            }
            return res.json();
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["admin-users"] });
            setIsEditDialogOpen(false);
            setSelectedUser(null);
            toast({ title: "User updated successfully" });
        },
        onError: (error: unknown) => {
            toast({
                title: "Failed to update user",
                description:
                    error instanceof Error ? error.message : "Unexpected error",
                variant: "destructive",
            });
        },
    });

    const createMutation = useMutation({
        mutationFn: async () => {
            const payload: Record<string, unknown> = {
                name: createForm.name,
                email: createForm.email,
                role: createForm.role,
                subscription_status: "active",
                is_active: true,
            };
            if (createForm.subscription_tier) {
                payload.subscription_tier = createForm.subscription_tier;
            }

            const res = await authFetch(
                `/api/admin/users?password=${encodeURIComponent(createForm.password)}`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                }
            );

            if (!res.ok) {
                throw new Error(await parseResponseError(res));
            }
            return res.json();
        },
        onSuccess: () => {
            toast({ title: "User created successfully" });
            queryClient.invalidateQueries({ queryKey: ["admin-users"] });
            closeCreateDialog();
        },
        onError: (error: unknown) => {
            toast({
                title: "Failed to create user",
                description:
                    error instanceof Error ? error.message : "Unexpected error",
                variant: "destructive",
            });
        },
    });

    const deleteMutation = useMutation({
        mutationFn: async (userId: string) => {
            const res = await authFetch(`/api/admin/users/${userId}`, {
                method: "DELETE",
            });
            if (!res.ok) {
                throw new Error(await parseResponseError(res));
            }
            return res.json();
        },
        onMutate: (userId: string) => {
            setDeletingUserId(userId);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["admin-users"] });
            toast({ title: "User disabled" });
        },
        onError: (error: unknown) => {
            toast({
                title: "Failed to disable user",
                description:
                    error instanceof Error ? error.message : "Unexpected error",
                variant: "destructive",
            });
        },
        onSettled: () => {
            setDeletingUserId(null);
        },
    });

    const handleDelete = (userId: string) => {
        if (!window.confirm("Disable this user?")) {
            return;
        }
        deleteMutation.mutate(userId);
    };

    const handleEdit = (user: User) => {
        setSelectedUser(user);
        setIsEditDialogOpen(true);
    };

    return (
        <Layout>
            <div className="space-y-6">
                <div className="flex justify-between items-center">
                    <h1 className="text-3xl font-bold tracking-tight">User Management</h1>
                    <Button onClick={openCreateDialog} tooltip="Create a new user account">Create User</Button>
                </div>

                <Card className="glass-card">
                    <CardHeader>
                        <CardTitle>All Users</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Name</TableHead>
                                    <TableHead>Email</TableHead>
                                    <TableHead>Role</TableHead>
                                    <TableHead>Subscription</TableHead>
                                    <TableHead>Tier</TableHead>
                                    <TableHead>Joined</TableHead>
                                    <TableHead>Overage Bonus</TableHead>
                                    <TableHead className="text-right">Actions</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {isLoading ? (
                                    <TableRow>
                                        <TableCell colSpan={8} className="text-center">
                                            Loading...
                                        </TableCell>
                                    </TableRow>
                                ) : (
                                    data?.users.map((user) => {
                                        const tierLabel =
                                            planNameById[user.subscription_tier] ||
                                            user.subscription_tier ||
                                            "N/A";
                                        const statusBadge = user.is_active
                                            ? user.subscription_status === "active"
                                                ? {
                                                      variant: "default" as const,
                                                      className:
                                                          "bg-green-500/15 text-green-500 border-green-500/20",
                                                  }
                                                : {
                                                      variant: "outline" as const,
                                                      className:
                                                          "text-amber-500 border-amber-500/40 bg-amber-500/10",
                                                  }
                                            : { variant: "destructive" as const, className: undefined };
                                        return (
                                            <TableRow key={user.id}>
                                                <TableCell className="font-medium">{user.name}</TableCell>
                                                <TableCell>{user.email}</TableCell>
                                                <TableCell>
                                                    <Badge
                                                        variant={
                                                            user.role === "admin"
                                                                ? "default"
                                                                : "secondary"
                                                        }
                                                    >
                                                        {user.role}
                                                    </Badge>
                                                </TableCell>
                                                <TableCell>
                                                    <Badge
                                                        variant={statusBadge.variant}
                                                        className={statusBadge.className}
                                                    >
                                                        {user.is_active
                                                            ? user.subscription_status
                                                            : "Disabled"}
                                                    </Badge>
                                                </TableCell>
                                                <TableCell className="capitalize">{tierLabel}</TableCell>
                                                <TableCell>
                                                    {new Date(user.created_at).toLocaleDateString()}
                                                </TableCell>
                                                <TableCell>{(user.quota_overage_bonus || 0).toLocaleString()}</TableCell>
                                                <TableCell className="text-right">
                                                    <div className="flex justify-end gap-2">
                                                        <Button
                                                            variant="ghost"
                                                            size="sm"
                                                            onClick={() => handleEdit(user)}
                                                            tooltip="Edit user details"
                                                        >
                                                            Edit
                                                        </Button>
                                                        <Button
                                                            variant="destructive"
                                                            size="sm"
                                                            onClick={() => handleDelete(user.id)}
                                                            disabled={
                                                                (!user.is_active) ||
                                                                (deleteMutation.isPending &&
                                                                    deletingUserId === user.id)
                                                            }
                                                            tooltip="Delete this user account"
                                                        >
                                                            {deleteMutation.isPending &&
                                                            deletingUserId === user.id
                                                                ? "Disabling..."
                                                                : user.is_active
                                                                    ? "Disable"
                                                                    : "Disabled"}
                                                        </Button>
                                                    </div>
                                                </TableCell>
                                            </TableRow>
                                        );
                                    })
                                )}
                            </TableBody>
                        </Table>
                    </CardContent>
                </Card>
            </div>

            <Dialog
                open={isEditDialogOpen}
                onOpenChange={(open) => {
                    setIsEditDialogOpen(open);
                    if (!open) {
                        setSelectedUser(null);
                    }
                }}
            >
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Edit User: {selectedUser?.name}</DialogTitle>
                        <DialogDescription>
                            Update role, subscription status, plan assignments, and free quota overage for this user.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-4">
                        <div className="space-y-2">
                            <Label>Role</Label>
                            <Select
                                value={selectedUser?.role ?? "customer"}
                                onValueChange={(val) =>
                                    setSelectedUser((prev) =>
                                        prev ? { ...prev, role: val } : prev
                                    )
                                }
                            >
                                <SelectTrigger>
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="customer">Customer</SelectItem>
                                    <SelectItem value="admin">Admin</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                        <div className="space-y-2">
                            <Label>Subscription Status</Label>
                            <Select
                                value={selectedUser?.subscription_status ?? "inactive"}
                                onValueChange={(val) =>
                                    setSelectedUser((prev) =>
                                        prev ? { ...prev, subscription_status: val } : prev
                                    )
                                }
                            >
                                <SelectTrigger>
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="active">Active</SelectItem>
                                    <SelectItem value="inactive">Inactive</SelectItem>
                                    <SelectItem value="past_due">Past Due</SelectItem>
                                    <SelectItem value="canceled">Canceled</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                        {selectedUser?.role === "customer" && (
                            <>
                            <div className="space-y-2">
                                <Label>Subscription Tier</Label>
                                <Select
                                    value={selectedUser?.subscription_tier || ""}
                                    onValueChange={(val) =>
                                        setSelectedUser((prev) =>
                                            prev ? { ...prev, subscription_tier: val } : prev
                                        )
                                    }
                                    disabled={!plans?.length}
                                >
                                    <SelectTrigger>
                                        <SelectValue placeholder="Select a plan" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {plans?.length ? (
                                            plans.map((plan) => (
                                                <SelectItem key={plan.id} value={plan.id}>
                                                    {plan.name}
                                                </SelectItem>
                                            ))
                                        ) : (
                                            <SelectItem value="" disabled>
                                                No plans available
                                            </SelectItem>
                                        )}
                                    </SelectContent>
                                </Select>
                                {!selectedUser?.subscription_tier && (
                                    <p className="text-xs text-destructive">
                                        Plan selection is required for customers.
                                    </p>
                                )}
                            </div>
                            <div className="space-y-2">
                                <Label>Free API Quota Overage (calls/mo)</Label>
                                <Input
                                    type="number"
                                    min={0}
                                    value={selectedUser?.quota_overage_bonus ?? 0}
                                    onChange={(event) =>
                                        setSelectedUser((prev) =>
                                            prev
                                                ? { ...prev, quota_overage_bonus: parseInt(event.target.value, 10) || 0 }
                                                : prev
                                        )
                                    }
                                />
                            </div>
                            </>
                        )}
                    </div>
                    <DialogFooter>
                        <Button
                            variant="outline"
                            onClick={() => {
                                setIsEditDialogOpen(false);
                                setSelectedUser(null);
                            }}
                            tooltip="Cancel editing and discard changes"
                        >
                            Cancel
                        </Button>
                        <Button
                            onClick={() => updateMutation.mutate({
                                role: selectedUser?.role,
                                subscription_status: selectedUser?.subscription_status,
                                subscription_tier: selectedUser?.subscription_tier,
                                quota_overage_bonus: selectedUser?.quota_overage_bonus ?? 0,
                            })}
                            disabled={
                                updateMutation.isPending ||
                                (selectedUser?.role === "customer" &&
                                    !selectedUser?.subscription_tier)
                            }
                            tooltip={updateMutation.isPending ? "Saving changes..." : "Save user changes"}
                        >
                            {updateMutation.isPending ? "Saving..." : "Save Changes"}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
            <Dialog
                open={isCreateDialogOpen}
                onOpenChange={(open) => {
                    if (!open) {
                        closeCreateDialog();
                    }
                }}
            >
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Create User</DialogTitle>
                        <DialogDescription>
                            Provide account details and assign a role for the new user.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-4">
                        <div className="space-y-2">
                            <Label>Name</Label>
                            <Input
                                value={createForm.name}
                                onChange={(event) =>
                                    setCreateForm((prev) => ({
                                        ...prev,
                                        name: event.target.value,
                                    }))
                                }
                            />
                        </div>
                        <div className="space-y-2">
                            <Label>Email</Label>
                            <Input
                                type="email"
                                value={createForm.email}
                                onChange={(event) =>
                                    setCreateForm((prev) => ({
                                        ...prev,
                                        email: event.target.value,
                                    }))
                                }
                            />
                        </div>
                        <div className="space-y-2">
                            <Label>Password</Label>
                            <Input
                                type="password"
                                value={createForm.password}
                                onChange={(event) =>
                                    setCreateForm((prev) => ({
                                        ...prev,
                                        password: event.target.value,
                                    }))
                                }
                            />
                        </div>
                        <div className="space-y-2">
                            <Label>Role</Label>
                            <Select
                                value={createForm.role}
                                onValueChange={(val) =>
                                    setCreateForm((prev) => ({
                                        ...prev,
                                        role: val as CreateUserForm["role"],
                                    }))
                                }
                            >
                                <SelectTrigger>
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="customer">Customer</SelectItem>
                                    <SelectItem value="admin">Admin</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                        {createForm.role === "customer" && (
                            <div className="space-y-2">
                                <Label>Subscription Tier</Label>
                                <Select
                                    value={createForm.subscription_tier}
                                    onValueChange={(val) =>
                                        setCreateForm((prev) => ({
                                            ...prev,
                                            subscription_tier: val,
                                        }))
                                    }
                                    disabled={!plans?.length}
                                >
                                    <SelectTrigger>
                                        <SelectValue placeholder="Select a plan" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {plans?.length ? (
                                            plans.map((plan) => (
                                                <SelectItem key={plan.id} value={plan.id}>
                                                    {plan.name}
                                                </SelectItem>
                                            ))
                                        ) : (
                                            <SelectItem value="" disabled>
                                                No plans available
                                            </SelectItem>
                                        )}
                                    </SelectContent>
                                </Select>
                                {!createForm.subscription_tier && (
                                    <p className="text-xs text-destructive">
                                        Plan selection is required for customers.
                                    </p>
                                )}
                            </div>
                        )}
                    </div>
                    <DialogFooter>
                        <Button variant="outline" onClick={closeCreateDialog} tooltip="Cancel user creation and close dialog">
                            Cancel
                        </Button>
                        <Button
                            onClick={() => createMutation.mutate()}
                            disabled={
                                createMutation.isPending ||
                                !createForm.email ||
                                !createForm.password ||
                                (createForm.role === "customer" &&
                                    !createForm.subscription_tier)
                            }
                            tooltip={createMutation.isPending ? "Creating user..." : "Create new user account"}
                        >
                            {createMutation.isPending ? "Creating..." : "Create User"}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </Layout>
    );
}
