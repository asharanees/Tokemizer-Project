import { Loader2 } from "lucide-react";

export function LoadingOverlay({ message = "Loading..." }: { message?: string }) {
    return (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-background/60 backdrop-blur-[2px] rounded-lg animate-in fade-in duration-200">
            <Loader2 className="h-8 w-8 text-primary animate-spin" />
            <p className="mt-2 text-sm font-medium text-muted-foreground">{message}</p>
        </div>
    );
}

export function LoadingSpinner({ className }: { className?: string }) {
    return <Loader2 className={`animate-spin ${className}`} />;
}
