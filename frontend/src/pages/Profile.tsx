import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Loader2 } from "lucide-react";
import { authFetch } from "@/lib/authFetch";

interface UserProfile {
    name: string;
    phone_number?: string;
}

const formatDate = (dateString: string): string => {
    try {
        const date = new Date(dateString);
        if (isNaN(date.getTime())) {
            return "Unknown date";
        }
        return date.toLocaleDateString();
    } catch {
        return "Unknown date";
    }
};

export default function Profile() {
    const { user, updateUser } = useAuth();
    const { toast } = useToast();
    const [isEditing, setIsEditing] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [formData, setFormData] = useState<UserProfile>({
        name: user?.name || "",
        phone_number: user?.phone_number || "",
    });

    if (!user) return null;

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleSave = async () => {
        if (!formData.name.trim()) {
            toast({
                title: "Error",
                description: "Name cannot be empty",
                variant: "destructive"
            });
            return;
        }

        setIsSaving(true);
        try {
            const res = await authFetch("/api/auth/profile", {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(formData),
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Failed to update profile");
            }

            const updatedUser = await res.json();
            updateUser(updatedUser);
            setIsEditing(false);
            toast({
                title: "Success",
                description: "Profile updated successfully",
            });
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : "Failed to update profile";
            toast({
                title: "Error",
                description: message,
                variant: "destructive"
            });
        } finally {
            setIsSaving(false);
        }
    };

    const handleCancel = () => {
        setFormData({
            name: user?.name || "",
            phone_number: user?.phone_number || "",
        });
        setIsEditing(false);
    };

    return (
        <Layout>
            <div className="space-y-6 max-w-4xl w-full">
                <h1 className="text-2xl sm:text-3xl font-bold tracking-tight font-display text-glow">Profile</h1>

                <Card className="glass-card">
                    <CardHeader>
                        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                            <div>
                                <CardTitle className="text-lg lg:text-base">Personal Information</CardTitle>
                                <CardDescription className="text-xs sm:text-sm">Manage your personal details.</CardDescription>
                            </div>
                            {!isEditing && (
                                <Button onClick={() => setIsEditing(true)} size="sm" className="w-full sm:w-auto" tooltip="Edit your profile information">
                                    Edit Profile
                                </Button>
                            )}
                        </div>
                    </CardHeader>
                    <CardContent className="space-y-4 lg:space-y-6">
                        <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-4">
                            <Avatar className="h-12 sm:h-16 w-12 sm:w-16">
                                <AvatarImage src={`https://avatar.vercel.sh/${user.email}`} />
                                <AvatarFallback>{user.name?.charAt(0) || "U"}</AvatarFallback>
                            </Avatar>
                            <div>
                                <h3 className="text-base sm:text-lg font-medium">{user.name}</h3>
                                <p className="text-xs sm:text-sm text-muted-foreground">{user.email}</p>
                            </div>
                        </div>

                        <div className="grid gap-4 md:grid-cols-2">
                            <div className="space-y-2">
                                <Label>Full Name</Label>
                                {isEditing ? (
                                    <Input
                                        name="name"
                                        value={formData.name}
                                        onChange={handleInputChange}
                                    />
                                ) : (
                                    <Input value={user.name} disabled />
                                )}
                            </div>
                            <div className="space-y-2">
                                <Label>Email Address</Label>
                                <Input value={user.email} disabled />
                            </div>
                            <div className="space-y-2">
                                <Label>Phone Number</Label>
                                {isEditing ? (
                                    <Input
                                        name="phone_number"
                                        type="tel"
                                        value={formData.phone_number || ""}
                                        onChange={handleInputChange}
                                        placeholder="+1 (555) 000-0000"
                                    />
                                ) : (
                                    <Input value={user?.phone_number || "Not provided"} disabled />
                                )}
                            </div>
                            <div className="space-y-2">
                                <Label>Member Since</Label>
                                <Input value={formatDate(user.created_at)} disabled />
                            </div>
                            <div className="space-y-2 md:col-span-2">
                                <Label>Free API Quota Overage</Label>
                                <Input value={`${(user.quota_overage_bonus || 0).toLocaleString()} bonus calls/month`} disabled />
                            </div>
                        </div>

                        {isEditing && (
                            <div className="flex gap-2 pt-4">
                                <Button onClick={handleSave} disabled={isSaving} tooltip={isSaving ? "Saving changes..." : "Save profile changes"}>
                                    {isSaving && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                                    Save Changes
                                </Button>
                                <Button variant="outline" onClick={handleCancel} disabled={isSaving} tooltip="Cancel editing and discard changes">
                                    Cancel
                                </Button>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </Layout>
    );
}
