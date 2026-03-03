import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
    AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { ReactNode } from "react";

interface ConfirmDialogProps {
    trigger?: ReactNode;
    title: string;
    description: string;
    onConfirm: () => void;
    confirmText?: string;
    cancelText?: string;
    variant?: "default" | "destructive";
    open?: boolean;
    onOpenChange?: (open: boolean) => void;
}

export function ConfirmDialog({
    trigger,
    title,
    description,
    onConfirm,
    confirmText = "Confirm",
    cancelText = "Cancel",
    variant = "default",
    open,
    onOpenChange,
}: ConfirmDialogProps) {
    return (
        <AlertDialog open={open} onOpenChange={onOpenChange}>
            {trigger && (
                <AlertDialogTrigger asChild>
                    {trigger}
                </AlertDialogTrigger>
            )}
            <AlertDialogContent>
                <AlertDialogHeader>
                    <AlertDialogTitle>{title}</AlertDialogTitle>
                    <AlertDialogDescription>
                        {description}
                    </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                    <AlertDialogCancel>{cancelText}</AlertDialogCancel>
                    <AlertDialogAction
                        onClick={onConfirm}
                        className={variant === "destructive" ? "bg-destructive text-destructive-foreground hover:bg-destructive/90" : ""}
                    >
                        {confirmText}
                    </AlertDialogAction>
                </AlertDialogFooter>
            </AlertDialogContent>
        </AlertDialog>
    );
}
