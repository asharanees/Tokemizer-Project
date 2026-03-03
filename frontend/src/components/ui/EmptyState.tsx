import { LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

interface EmptyStateProps {
    icon: LucideIcon;
    title: string;
    description: string;
    actionText?: string;
    onAction?: () => void;
    actionTooltip?: string;
}

export function EmptyState({
    icon: Icon,
    title,
    description,
    actionText,
    onAction,
    actionTooltip,
}: EmptyStateProps) {
    return (
        <div className="flex flex-col items-center justify-center p-12 text-center space-y-4 rounded-xl border-2 border-dashed border-border/50 bg-muted/5">
            <div className="p-4 rounded-full bg-primary/5">
                <Icon className="h-10 w-10 text-primary/40" />
            </div>
            <div className="max-w-[300px] space-y-1">
                <h3 className="font-semibold text-lg">{title}</h3>
                <p className="text-sm text-muted-foreground">{description}</p>
            </div>
            {actionText && onAction && (
                <Button onClick={onAction} variant="outline" className="mt-2" tooltip={actionTooltip}>
                    {actionText}
                </Button>
            )}
        </div>
    );
}
